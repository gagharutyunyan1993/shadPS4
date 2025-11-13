// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <xxhash.h>
#include "common/config.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_shader_cache.h"

namespace Vulkan {

// Cache file format version
constexpr u32 CACHE_FILE_MAGIC = 0x53505653; // "SPVS" in hex
constexpr u32 CACHE_FILE_VERSION = 1;

struct CacheFileHeader {
    u32 magic;
    u32 version;
    u32 spv_size; // Size in u32 words
};

struct MetadataHeader {
    u32 magic;
    u32 version;
    u32 cache_version;
};

ShaderCache::ShaderCache(const std::filesystem::path& cache_dir, u32 cache_version)
    : cache_dir(cache_dir), cache_version(cache_version) {

    // Check if shader caching is enabled in config
    if (!Config::shaderCacheEnabled()) {
        enabled = false;
        LOG_INFO(Render_Vulkan, "Shader cache is disabled in config");
        return;
    }

    // Create cache directory structure
    spirv_dir = cache_dir / "spirv";
    metadata_path = cache_dir / "metadata.bin";

    try {
        std::filesystem::create_directories(spirv_dir);
    } catch (const std::filesystem::filesystem_error& e) {
        LOG_ERROR(Render_Vulkan, "Failed to create shader cache directory: {}", e.what());
        enabled = false;
        return;
    }

    // Validate existing cache or clear if invalid
    if (!ValidateCache()) {
        LOG_WARNING(Render_Vulkan, "Shader cache validation failed, clearing cache");
        Clear();
        SaveMetadata();
    }

    LOG_INFO(Render_Vulkan, "Shader cache initialized at: {}", cache_dir.string());
}

ShaderCache::~ShaderCache() {
    SaveMetadata();

    LOG_INFO(Render_Vulkan,
             "Shader cache statistics - Hits: {}, Misses: {}, Loads: {}, Saves: {}, Size: {} KB",
             stats.hits, stats.misses, stats.loads, stats.saves, stats.total_size / 1024);
}

std::optional<std::vector<u32>> ShaderCache::Load(const CacheKey& key) {
    if (!enabled) {
        return std::nullopt;
    }

    std::shared_lock lock(mutex);

    const auto cache_path = GetCachePath(key);
    if (!std::filesystem::exists(cache_path)) {
        stats.misses++;
        return std::nullopt;
    }

    try {
        Common::FS::IOFile file(cache_path, Common::FS::FileAccessMode::Read);
        if (!file.IsOpen()) {
            stats.misses++;
            return std::nullopt;
        }

        // Read header
        CacheFileHeader header;
        if (!file.ReadObject(header)) {
            LOG_WARNING(Render_Vulkan, "Failed to read cache file header: {}",
                       cache_path.filename().string());
            stats.misses++;
            return std::nullopt;
        }

        // Validate header
        if (header.magic != CACHE_FILE_MAGIC || header.version != CACHE_FILE_VERSION) {
            LOG_WARNING(Render_Vulkan, "Invalid cache file header: {}",
                       cache_path.filename().string());
            stats.misses++;
            return std::nullopt;
        }

        // Read SPIR-V data
        std::vector<u32> spv_code(header.spv_size);
        const size_t read_count = file.ReadSpan<u32>(spv_code);

        if (read_count != header.spv_size) {
            LOG_WARNING(Render_Vulkan, "Failed to read SPIR-V data from cache: {}",
                       cache_path.filename().string());
            stats.misses++;
            return std::nullopt;
        }

        stats.hits++;
        stats.loads++;

        return spv_code;

    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "Exception while loading shader cache: {}", e.what());
        stats.misses++;
        return std::nullopt;
    }
}

void ShaderCache::Save(const CacheKey& key, std::span<const u32> spv_code) {
    if (!enabled || spv_code.empty()) {
        return;
    }

    std::unique_lock lock(mutex);

    const auto cache_path = GetCachePath(key);

    try {
        Common::FS::IOFile file(cache_path, Common::FS::FileAccessMode::Create);
        if (!file.IsOpen()) {
            LOG_WARNING(Render_Vulkan, "Failed to create cache file: {}",
                       cache_path.filename().string());
            return;
        }

        // Write header
        CacheFileHeader header{
            .magic = CACHE_FILE_MAGIC,
            .version = CACHE_FILE_VERSION,
            .spv_size = static_cast<u32>(spv_code.size()),
        };

        if (!file.WriteObject(header)) {
            LOG_WARNING(Render_Vulkan, "Failed to write cache file header: {}",
                       cache_path.filename().string());
            return;
        }

        // Write SPIR-V data
        const size_t written = file.WriteSpan<u32>(spv_code);
        if (written != spv_code.size()) {
            LOG_WARNING(Render_Vulkan, "Failed to write SPIR-V data to cache: {}",
                       cache_path.filename().string());
            return;
        }

        file.Flush();

        stats.saves++;
        stats.total_size += sizeof(header) + (spv_code.size() * sizeof(u32));

    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "Exception while saving shader cache: {}", e.what());
    }
}

void ShaderCache::Clear() {
    if (!enabled) {
        return;
    }

    std::unique_lock lock(mutex);

    try {
        if (std::filesystem::exists(spirv_dir)) {
            // Remove all cache files
            for (const auto& entry : std::filesystem::directory_iterator(spirv_dir)) {
                std::filesystem::remove(entry.path());
            }
            LOG_INFO(Render_Vulkan, "Shader cache cleared");
        }

        // Reset statistics
        stats = CacheStats{};

    } catch (const std::filesystem::filesystem_error& e) {
        LOG_ERROR(Render_Vulkan, "Failed to clear shader cache: {}", e.what());
    }
}

std::filesystem::path ShaderCache::GetCachePath(const CacheKey& key) const {
    const auto hash = HashKey(key);
    return spirv_dir / fmt::format("{:016X}.spv", hash);
}

u64 ShaderCache::HashKey(const CacheKey& key) const {
    // Use XXH3 for fast and high-quality hashing
    XXH3_state_t state;
    XXH3_64bits_reset(&state);
    XXH3_64bits_update(&state, &key.shader_hash, sizeof(key.shader_hash));
    XXH3_64bits_update(&state, &key.runtime_hash, sizeof(key.runtime_hash));
    XXH3_64bits_update(&state, &key.permutation_idx, sizeof(key.permutation_idx));
    return XXH3_64bits_digest(&state);
}

bool ShaderCache::ValidateCache() {
    if (!std::filesystem::exists(metadata_path)) {
        return false;
    }

    return LoadMetadata();
}

void ShaderCache::SaveMetadata() {
    if (!enabled) {
        return;
    }

    try {
        Common::FS::IOFile file(metadata_path, Common::FS::FileAccessMode::Create);
        if (!file.IsOpen()) {
            LOG_WARNING(Render_Vulkan, "Failed to save cache metadata");
            return;
        }

        MetadataHeader metadata{
            .magic = CACHE_FILE_MAGIC,
            .version = CACHE_FILE_VERSION,
            .cache_version = cache_version,
        };

        file.WriteObject(metadata);
        file.Flush();

    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "Exception while saving cache metadata: {}", e.what());
    }
}

bool ShaderCache::LoadMetadata() {
    try {
        Common::FS::IOFile file(metadata_path, Common::FS::FileAccessMode::Read);
        if (!file.IsOpen()) {
            return false;
        }

        MetadataHeader metadata;
        if (!file.ReadObject(metadata)) {
            return false;
        }

        // Validate metadata
        if (metadata.magic != CACHE_FILE_MAGIC ||
            metadata.version != CACHE_FILE_VERSION ||
            metadata.cache_version != cache_version) {
            LOG_WARNING(Render_Vulkan,
                       "Cache metadata mismatch - Magic: {:#x} (expected {:#x}), "
                       "Version: {} (expected {}), Cache version: {} (expected {})",
                       metadata.magic, CACHE_FILE_MAGIC,
                       metadata.version, CACHE_FILE_VERSION,
                       metadata.cache_version, cache_version);
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "Exception while loading cache metadata: {}", e.what());
        return false;
    }
}

} // namespace Vulkan
