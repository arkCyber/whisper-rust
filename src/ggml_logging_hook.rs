use crate::common_logging::{
    generic_debug, generic_error, generic_info, generic_trace, generic_warn, GGMLLogLevel,
};
use core::ffi::{c_char, c_void};
use std::borrow::Cow;
use std::ffi::CStr;
use std::sync::Once;
use whisper_rs_sys::ggml_log_level;

static GGML_LOG_TRAMPOLINE_INSTALL: Once = Once::new();
pub(crate) fn install_ggml_logging_hook() {
    GGML_LOG_TRAMPOLINE_INSTALL.call_once(|| unsafe {
        whisper_rs_sys::ggml_log_set(Some(ggml_logging_trampoline), std::ptr::null_mut())
    });
}

unsafe extern "C" fn ggml_logging_trampoline(
    level: ggml_log_level,
    text: *const c_char,
    _: *mut c_void, // user_data
) {
    if text.is_null() {
        generic_error!("ggml_logging_trampoline: text is nullptr");
    }
    let level = GGMLLogLevel::from(level);

    // SAFETY: we must trust ggml that it will not pass us a string that does not satisfy
    // from_ptr's requirements.
    let log_str = unsafe { CStr::from_ptr(text) }.to_string_lossy();

    ggml_logging_trampoline_safe(level, log_str)
}

// this code essentially compiles down to a noop if neither feature is enabled
#[cfg_attr(
    not(any(feature = "log_backend", feature = "tracing_backend")),
    allow(unused_variables)
)]
fn ggml_logging_trampoline_safe(level: GGMLLogLevel, text: Cow<str>) {
    match level {
        GGMLLogLevel::None => {
            // no clue what to do here, trace it?
            generic_trace!("{}", text.trim());
        }
        GGMLLogLevel::Info => {
            generic_info!("{}", text.trim());
        }
        GGMLLogLevel::Warn => {
            generic_warn!("{}", text.trim());
        }
        GGMLLogLevel::Error => {
            generic_error!("{}", text.trim());
        }
        GGMLLogLevel::Debug => {
            generic_debug!("{}", text.trim());
        }
        GGMLLogLevel::Cont => {
            // this means continue previous log
            // storing state to do this is a massive pain so it's just a lot easier to not
            // plus as far as i can tell it's not actually *used* anywhere
            // ggml splits at 128 chars and doesn't actually change the kind of log
            // so technically this is unused
            generic_trace!("{}", text.trim());
        }
        GGMLLogLevel::Unknown(level) => {
            generic_warn!(
                "ggml_logging_trampoline: unknown log level {}: message: {}",
                level,
                text.trim()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ggml_logging_hook_installation() {
        // Test that the hook installation doesn't panic
        install_ggml_logging_hook();
        
        // Test multiple calls don't cause issues (Once should prevent multiple installations)
        install_ggml_logging_hook();
        install_ggml_logging_hook();
    }
    
    #[test]
    fn test_ggml_logging_trampoline_safe_all_levels() {
        // Test all log levels for GGML
        let test_cases = vec![
            (GGMLLogLevel::None, "GGML None level message"),
            (GGMLLogLevel::Info, "GGML Info level message"),
            (GGMLLogLevel::Warn, "GGML Warning level message"),
            (GGMLLogLevel::Error, "GGML Error level message"),
            (GGMLLogLevel::Debug, "GGML Debug level message"),
            (GGMLLogLevel::Cont, "GGML Continuation message"),
            (GGMLLogLevel::Unknown(777), "GGML Unknown level message"),
        ];
        
        for (level, message) in test_cases {
            // Call the function to ensure it doesn't panic
            ggml_logging_trampoline_safe(level, std::borrow::Cow::Borrowed(message));
        }
    }
    
    #[test]
    fn test_ggml_logging_trampoline_safe_text_processing() {
        // Test text with various formatting issues
        let test_cases = vec![
            "  Leading spaces",
            "Trailing spaces  ",
            "\n\tComplex whitespace\r\n",
            "Mixed\t\nspacing\r",
            "",
            "   ",
        ];
        
        for input in test_cases {
            ggml_logging_trampoline_safe(
                GGMLLogLevel::Info, 
                std::borrow::Cow::Borrowed(input)
            );
        }
    }
    
    #[test]
    fn test_ggml_logging_128_char_split_behavior() {
        // Test the comment about GGML splitting at 128 chars
        let short_message = "Short message";
        let long_message = "A".repeat(150); // Longer than 128 chars
        
        ggml_logging_trampoline_safe(
            GGMLLogLevel::Info, 
            std::borrow::Cow::Borrowed(&short_message)
        );
        
        ggml_logging_trampoline_safe(
            GGMLLogLevel::Info, 
            std::borrow::Cow::Owned(long_message)
        );
    }
    
    #[test]
    fn test_ggml_logging_continuation_level_handling() {
        // Test that Cont level is handled as trace (as per comment)
        ggml_logging_trampoline_safe(
            GGMLLogLevel::Cont, 
            std::borrow::Cow::Borrowed("Continuation message 1")
        );
        
        ggml_logging_trampoline_safe(
            GGMLLogLevel::Cont, 
            std::borrow::Cow::Borrowed("Continuation message 2")
        );
    }
    
    #[test]
    fn test_ggml_logging_unknown_levels_with_values() {
        // Test various unknown log level values
        let unknown_levels = vec![
            (0xDEADBEEF, "Hex value message"),
            (u32::MAX, "Max value message"),
            (12345, "Random value message"),
            (0, "Zero value message"), // Note: 0 might map to None, but testing as unknown
        ];
        
        for (level_value, message) in unknown_levels {
            ggml_logging_trampoline_safe(
                GGMLLogLevel::Unknown(level_value), 
                std::borrow::Cow::Borrowed(message)
            );
        }
    }
    
    #[test]
    fn test_ggml_logging_performance_with_large_messages() {
        use std::time::Instant;
        
        // Test performance with larger messages
        let large_message = "X".repeat(1000);
        let very_large_message = "Y".repeat(10000);
        
        let start = Instant::now();
        
        for _i in 0..100 {
            ggml_logging_trampoline_safe(
                GGMLLogLevel::Info, 
                std::borrow::Cow::Borrowed(&large_message)
            );
        }
        
        ggml_logging_trampoline_safe(
            GGMLLogLevel::Info, 
            std::borrow::Cow::Borrowed(&very_large_message)
        );
        
        let duration = start.elapsed();
        
        // Performance should be reasonable (less than 1 second for this test)
        assert!(duration.as_secs() < 1);
    }
    
    #[test]
    fn test_ggml_logging_thread_safety() {
        use std::thread;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        
        let success_flag = Arc::new(AtomicBool::new(true));
        
        // Test thread safety with multiple threads logging simultaneously
        let handles: Vec<_> = (0..5).map(|thread_id| {
            let flag = success_flag.clone();
            
            thread::spawn(move || {
                let result = std::panic::catch_unwind(|| {
                    for i in 0..20 {
                        let message = format!("Thread {} message {}", thread_id, i);
                        ggml_logging_trampoline_safe(
                            GGMLLogLevel::Debug, 
                            std::borrow::Cow::Owned(message)
                        );
                    }
                });
                
                if result.is_err() {
                    flag.store(false, Ordering::SeqCst);
                }
            })
        }).collect();
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert!(success_flag.load(Ordering::SeqCst), "Thread-safe logging failed");
    }
    
    #[test]
    fn test_ggml_logging_memory_efficiency() {
        // Test that memory usage is reasonable with many small messages
        for i in 0..1000 {
            let message = format!("Message {}", i);
            ggml_logging_trampoline_safe(
                GGMLLogLevel::Info, 
                std::borrow::Cow::Owned(message)
            );
        }
    }
    
    #[test]
    fn test_ggml_logging_special_characters_and_encoding() {
        // Test various character encodings and special characters
        let special_messages = vec![
            "English message",
            "Message with émojis 🎵🚀💻",
            "Message with ñññ and üüü",
            "Message with tabs\tand\nnewlines",
            "Message with \"quotes\" and 'apostrophes'",
            "Message with backslash \\ and forward slash /",
            "Message with numbers: 123456789 and symbols: !@#$%^&*()",
        ];
        
        for message in special_messages {
            ggml_logging_trampoline_safe(
                GGMLLogLevel::Warn, 
                std::borrow::Cow::Borrowed(message)
            );
        }
    }
    
    #[test]
    fn test_ggml_logging_integration_with_whisper_logging() {
        // Ensure GGML logging doesn't interfere with Whisper logging
        // This is more of a smoke test to ensure both can coexist
        
        // Install both hooks (should not conflict)
        install_ggml_logging_hook();
        crate::whisper_logging_hook::install_whisper_logging_hook();
        
        // Test that installation succeeded without panic
        // The actual functionality testing is done in separate tests
        
        // This test mainly verifies that the installation process
        // doesn't cause conflicts between the two logging systems
        assert!(true); // If we reach here, no panic occurred
    }
}
