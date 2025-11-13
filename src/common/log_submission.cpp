// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "log_submission.h"
#include "common/config.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/scm_rev.h"
#include "common/version.h"
#include <fstream>
#include <sstream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <sysinfoapi.h>
#elif __APPLE__
#include <sys/sysctl.h>
#include <sys/utsname.h>
#else
#include <sys/utsname.h>
#include <fstream>
#endif

namespace LogSubmission {

static std::string GetCPUModel() {
#ifdef _WIN32
    char cpu_brand[0x40];
    DWORD buffer_size = sizeof(cpu_brand);
    HKEY hkey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hkey) == ERROR_SUCCESS) {
        RegQueryValueExA(hkey, "ProcessorNameString", nullptr, nullptr,
                        reinterpret_cast<LPBYTE>(cpu_brand), &buffer_size);
        RegCloseKey(hkey);
        return std::string(cpu_brand);
    }
    return "Unknown CPU";
#elif __APPLE__
    char cpu_brand[256];
    size_t size = sizeof(cpu_brand);
    if (sysctlbyname("machdep.cpu.brand_string", &cpu_brand, &size, nullptr, 0) == 0) {
        return std::string(cpu_brand);
    }
    return "Unknown CPU";
#else
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                return line.substr(pos + 2);
            }
        }
    }
    return "Unknown CPU";
#endif
}

static std::string GetOSName() {
#ifdef _WIN32
    return "Windows";
#elif __APPLE__
    return "macOS";
#elif __linux__
    return "Linux";
#elif __FreeBSD__
    return "FreeBSD";
#else
    return "Unknown OS";
#endif
}

static std::string GetOSVersion() {
#ifdef _WIN32
    OSVERSIONINFOEX info;
    ZeroMemory(&info, sizeof(OSVERSIONINFOEX));
    info.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

    if (GetVersionEx((LPOSVERSIONINFO)&info)) {
        return std::to_string(info.dwMajorVersion) + "." +
               std::to_string(info.dwMinorVersion) + "." +
               std::to_string(info.dwBuildNumber);
    }
    return "Unknown";
#else
    struct utsname buf;
    if (uname(&buf) == 0) {
        return std::string(buf.release);
    }
    return "Unknown";
#endif
}

static std::string GetCPUCores() {
    return std::to_string(std::thread::hardware_concurrency());
}

SystemInfo CollectSystemInfo(const std::string& game_id) {
    SystemInfo info;

    info.os_name = GetOSName();
    info.os_version = GetOSVersion();
    info.cpu_model = GetCPUModel();
    info.cpu_cores = GetCPUCores();
    info.gpu_model = "N/A"; // TODO: Add GPU detection
    info.emulator_version = std::string(Common::g_scm_desc);
    info.game_id = game_id;

    return info;
}

std::vector<u8> CompressLogFiles() {
    std::vector<u8> result;

    try {
        const auto log_dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
        const auto log_file = log_dir / "shad_log.txt";

        // For now, just read the main log file without compression
        // TODO: Implement proper ZIP compression with multiple log files
        if (std::filesystem::exists(log_file)) {
            std::ifstream file(log_file, std::ios::binary);
            if (file) {
                file.seekg(0, std::ios::end);
                size_t size = file.tellg();
                file.seekg(0, std::ios::beg);

                // Limit log size to 10MB to prevent huge uploads
                const size_t max_size = 10 * 1024 * 1024;
                if (size > max_size) {
                    // Read only the last 10MB
                    file.seekg(size - max_size, std::ios::beg);
                    size = max_size;
                }

                result.resize(size);
                file.read(reinterpret_cast<char*>(result.data()), size);
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR(Common, "Failed to collect log files: {}", e.what());
    }

    return result;
}

LogData PrepareLogData(const std::string& game_id,
                       const std::string& error_context,
                       bool is_crash) {
    LogData data;

    data.system_info = CollectSystemInfo(game_id);
    data.compressed_logs = CompressLogFiles();
    data.error_context = error_context;
    data.is_crash = is_crash;

    return data;
}

bool SubmitLogs(const LogData& log_data) {
    // Check if log submission is enabled
    if (!Config::getEnableAutoLogSubmission()) {
        LOG_INFO(Common, "Log submission is disabled in config");
        return false;
    }

    // Get the submission endpoint
    std::string endpoint = Config::getLogSubmissionEndpoint();
    if (endpoint.empty()) {
        LOG_WARNING(Common, "Log submission endpoint is not configured");
        return false;
    }

    LOG_INFO(Common, "Preparing to submit logs to: {}", endpoint);
    LOG_INFO(Common, "System Info: OS={} {}, CPU={} ({}), Emulator={}",
             log_data.system_info.os_name,
             log_data.system_info.os_version,
             log_data.system_info.cpu_model,
             log_data.system_info.cpu_cores,
             log_data.system_info.emulator_version);
    LOG_INFO(Common, "Log size: {} bytes", log_data.compressed_logs.size());

    // TODO: Implement actual HTTP submission
    // For now, this is a placeholder that logs what would be submitted
    // When a backend server is ready, implement HTTP POST with:
    // - Multipart form data containing:
    //   - system_info (JSON)
    //   - log_file (binary)
    //   - error_context (text)
    //   - is_crash (boolean)
    //   - game_id (text)

    LOG_WARNING(Common, "HTTP submission not yet implemented - log data prepared but not sent");
    LOG_INFO(Common, "To implement submission, add HTTP client library (e.g., libcurl)");

    return false; // Return false until actual implementation is complete
}

bool TriggerManualSubmission(const std::string& game_id) {
    if (!Config::getEnableAutoLogSubmission()) {
        LOG_INFO(Common, "Cannot submit logs: Log submission is disabled in config");
        return false;
    }

    LOG_INFO(Common, "Manual log submission triggered");

    LogData data = PrepareLogData(game_id, "Manual submission", false);
    return SubmitLogs(data);
}

void OnCrashSubmission(const std::string& game_id, const std::string& error_context) {
    // Only submit if both auto log submission and auto submit on crash are enabled
    if (!Config::getEnableAutoLogSubmission()) {
        return;
    }

    if (!Config::getAutoSubmitOnCrash()) {
        LOG_INFO(Common, "Crash detected but automatic crash submission is disabled");
        return;
    }

    LOG_CRITICAL(Common, "Crash detected - attempting automatic log submission");

    LogData data = PrepareLogData(game_id, error_context, true);
    SubmitLogs(data);
}

} // namespace LogSubmission
