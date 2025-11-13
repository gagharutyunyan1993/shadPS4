// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/assert.h"
#include "common/log_submission.h"
#include "common/logging/backend.h"

#if defined(ARCH_X86_64)
#define Crash() __asm__ __volatile__("int $3")
#elif defined(ARCH_ARM64)
#define Crash() __asm__ __volatile__("brk 0")
#else
#error "Missing Crash() implementation for target CPU architecture."
#endif

void assert_fail_impl() {
    Common::Log::Stop();
    std::fflush(stdout);
    Crash();
}

[[noreturn]] void unreachable_impl() {
    LOG_CRITICAL(Debug, "Unreachable code executed");

    // Attempt automatic log submission if enabled
    LogSubmission::OnCrashSubmission("", "Unreachable code executed");

    Common::Log::Stop();
    std::fflush(stdout);
    Crash();
    throw std::runtime_error("Unreachable code");
}

void assert_fail_debug_msg(const char* msg) {
    LOG_CRITICAL(Debug, "Assertion Failed!\n{}", msg);

    // Attempt automatic log submission if enabled
    LogSubmission::OnCrashSubmission("", std::string("Assertion Failed: ") + msg);

    assert_fail_impl();
}
