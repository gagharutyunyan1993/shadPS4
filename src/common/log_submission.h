// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include "types.h"

namespace LogSubmission {

struct SystemInfo {
    std::string os_name;
    std::string os_version;
    std::string cpu_model;
    std::string cpu_cores;
    std::string gpu_model;
    std::string emulator_version;
    std::string game_id;
};

struct LogData {
    std::vector<u8> compressed_logs;    // ZIP compressed log files
    SystemInfo system_info;
    std::string error_context;           // Optional context about the error
    bool is_crash;                       // Whether this was triggered by a crash
};

// Collect system information
SystemInfo CollectSystemInfo(const std::string& game_id = "");

// Collect and compress log files
std::vector<u8> CompressLogFiles();

// Prepare complete log data for submission
LogData PrepareLogData(const std::string& game_id = "",
                       const std::string& error_context = "",
                       bool is_crash = false);

// Submit log data to the configured endpoint
bool SubmitLogs(const LogData& log_data);

// Manual submission trigger (returns true if submission was successful)
bool TriggerManualSubmission(const std::string& game_id = "");

// Automatic submission on crash (only if enabled in config)
void OnCrashSubmission(const std::string& game_id = "", const std::string& error_context = "");

} // namespace LogSubmission
