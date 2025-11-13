// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <shared_mutex>
#include <vector>
#include <tsl/robin_map.h>
#include "common/types.h"

namespace Vulkan {

/**
 * Manages persistent caching of compiled SPIR-V shaders to disk
 * to reduce stuttering on subsequent game launches.
 */
class ShaderCache {
public:
    struct CacheKey {
        u64 shader_hash;        // Base shader hash from binary
        u64 runtime_hash;       // Hash of RuntimeInfo
        u64 permutation_idx;    // Permutation index

        bool operator==(const CacheKey& other) const {
            return shader_hash == other.shader_hash &&
                   runtime_hash == other.runtime_hash &&
                   permutation_idx == other.permutation_idx;
        }
    };

    struct CacheStats {
        u32 hits{0};            // Number of cache hits
        u32 misses{0};          // Number of cache misses
        u32 saves{0};           // Number of shaders saved
        u32 loads{0};           // Number of shaders loaded
        u64 total_size{0};      // Total cache size in bytes
    };

    /**
     * Constructs the shader cache and initializes cache directory.
     * @param cache_dir Path to the cache directory
     * @param cache_version Version number for cache invalidation
     */
    explicit ShaderCache(const std::filesystem::path& cache_dir, u32 cache_version);

    /**
     * Destructor - logs cache statistics.
     */
    ~ShaderCache();

    /**
     * Attempts to load a cached SPIR-V shader from disk.
     * @param key Cache key identifying the shader
     * @return SPIR-V bytecode if found, std::nullopt otherwise
     */
    std::optional<std::vector<u32>> Load(const CacheKey& key);

    /**
     * Saves a compiled SPIR-V shader to disk cache.
     * @param key Cache key identifying the shader
     * @param spv_code SPIR-V bytecode to save
     */
    void Save(const CacheKey& key, std::span<const u32> spv_code);

    /**
     * Clears all cached shaders from disk.
     */
    void Clear();

    /**
     * Returns cache statistics.
     */
    const CacheStats& GetStats() const {
        return stats;
    }

    /**
     * Checks if shader caching is enabled.
     */
    bool IsEnabled() const {
        return enabled;
    }

private:
    /**
     * Generates a cache file path from a cache key.
     */
    std::filesystem::path GetCachePath(const CacheKey& key) const;

    /**
     * Generates a unique hash from a cache key.
     */
    u64 HashKey(const CacheKey& key) const;

    /**
     * Validates cache version and GPU compatibility.
     * @return true if cache is valid, false if it should be cleared
     */
    bool ValidateCache();

    /**
     * Saves cache metadata (version, GPU info) to disk.
     */
    void SaveMetadata();

    /**
     * Loads cache metadata from disk.
     */
    bool LoadMetadata();

    std::filesystem::path cache_dir;
    std::filesystem::path spirv_dir;
    std::filesystem::path metadata_path;
    u32 cache_version;
    bool enabled{true};
    CacheStats stats;
    mutable std::shared_mutex mutex; // For thread-safe cache access
};

} // namespace Vulkan

// Hash specialization for CacheKey
template <>
struct std::hash<Vulkan::ShaderCache::CacheKey> {
    std::size_t operator()(const Vulkan::ShaderCache::CacheKey& key) const noexcept {
        std::size_t h1 = std::hash<u64>{}(key.shader_hash);
        std::size_t h2 = std::hash<u64>{}(key.runtime_hash);
        std::size_t h3 = std::hash<u64>{}(key.permutation_idx);
        // Combine hashes
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};
