use crate::common_logging::{
    generic_debug, generic_error, generic_info, generic_trace, generic_warn, GGMLLogLevel,
};
use core::ffi::{c_char, c_void};
use std::borrow::Cow;
use std::ffi::CStr;
use std::sync::Once;
use whisper_rs_sys::ggml_log_level;

static WHISPER_LOG_TRAMPOLINE_INSTALL: Once = Once::new();
pub(crate) fn install_whisper_logging_hook() {
    WHISPER_LOG_TRAMPOLINE_INSTALL.call_once(|| unsafe {
        whisper_rs_sys::whisper_log_set(Some(whisper_logging_trampoline), std::ptr::null_mut())
    });
}

unsafe extern "C" fn whisper_logging_trampoline(
    level: ggml_log_level,
    text: *const c_char,
    _: *mut c_void, // user_data
) {
    if text.is_null() {
        generic_error!("whisper_logging_trampoline: text is nullptr");
    }
    let level = GGMLLogLevel::from(level);

    // SAFETY: we must trust whisper.cpp that it will not pass us a string that does not satisfy
    // from_ptr's requirements.
    let log_str = unsafe { CStr::from_ptr(text) }.to_string_lossy();

    whisper_logging_trampoline_safe(level, log_str)
}

// this code essentially compiles down to a noop if neither feature is enabled
#[cfg_attr(
    not(any(feature = "log_backend", feature = "tracing_backend")),
    allow(unused_variables)
)]
fn whisper_logging_trampoline_safe(level: GGMLLogLevel, text: Cow<str>) {
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
            // whisper splits at 1024 chars and doesn't actually change the kind
            // so technically this is unused
            generic_trace!("{}", text.trim());
        }
        GGMLLogLevel::Unknown(level) => {
            generic_warn!(
                "whisper_logging_trampoline: unknown log level {}: message: {}",
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
    fn test_whisper_logging_hook_installation() {
        // Test that the hook installation doesn't panic
        install_whisper_logging_hook();
        
        // Test multiple calls don't cause issues
        install_whisper_logging_hook();
        install_whisper_logging_hook();
    }
    
    #[test]
    fn test_whisper_logging_trampoline_safe_basic() {
        // Test basic functionality of the safe trampoline function
        // This tests the function without actually capturing logs
        
        let test_cases = vec![
            (GGMLLogLevel::None, "None level message"),
            (GGMLLogLevel::Info, "Info level message"),
            (GGMLLogLevel::Warn, "Warning level message"),
            (GGMLLogLevel::Error, "Error level message"),
            (GGMLLogLevel::Debug, "Debug level message"),
            (GGMLLogLevel::Cont, "Continuation message"),
            (GGMLLogLevel::Unknown(999), "Unknown level message"),
        ];
        
        for (level, message) in test_cases {
            // Call the function to ensure it doesn't panic
            whisper_logging_trampoline_safe(level, std::borrow::Cow::Borrowed(message));
        }
    }
    
    #[test]
    fn test_whisper_logging_trampoline_safe_text_trimming() {
        // Test text trimming functionality
        let test_cases = vec![
            "  \n\tMessage with whitespace\t\n  ",
            "",
            "   \n\t   ",
            "Normal message",
            "Message with 🎵 unicode",
            "Message with\nnewlines\nand\ttabs",
        ];
        
        for message in test_cases {
            // Call the function to ensure it doesn't panic with various text formats
            whisper_logging_trampoline_safe(
                GGMLLogLevel::Info, 
                std::borrow::Cow::Borrowed(message)
            );
        }
    }
    
    #[test]
    fn test_whisper_logging_unknown_levels() {
        // Test various unknown log levels
        let unknown_levels = vec![42, 999, 0xFFFF, u32::MAX];
        
        for level_value in unknown_levels {
            whisper_logging_trampoline_safe(
                GGMLLogLevel::Unknown(level_value), 
                std::borrow::Cow::Borrowed(&format!("Message for level {}", level_value))
            );
        }
    }
    
    #[test]
    fn test_whisper_logging_cow_variants() {
        // Test both Borrowed and Owned Cow variants
        let borrowed_message = "Borrowed message";
        let owned_message = String::from("Owned message");
        
        whisper_logging_trampoline_safe(
            GGMLLogLevel::Info, 
            std::borrow::Cow::Borrowed(borrowed_message)
        );
        
        whisper_logging_trampoline_safe(
            GGMLLogLevel::Info, 
            std::borrow::Cow::Owned(owned_message)
        );
    }
    
    #[test]
    fn test_whisper_logging_special_characters() {
        // Test with special characters and unicode
        let test_messages = vec![
            "Message with 🎵 unicode",
            "Message with\nnewlines\nand\ttabs",
            "Message with special chars: !@#$%^&*()",
            "Message with quotes: \"quoted text\" and 'single quotes'",
            "Message with backslash \\ and forward slash /",
        ];
        
        for message in test_messages {
            whisper_logging_trampoline_safe(
                GGMLLogLevel::Info, 
                std::borrow::Cow::Borrowed(message)
            );
        }
    }
    
    #[test]
    fn test_whisper_logging_error_handling() {
        // Test various error scenarios that should be handled gracefully
        let error_cases = vec![
            (GGMLLogLevel::Error, "Critical error occurred"),
            (GGMLLogLevel::Warn, "Warning: something went wrong"),
            (GGMLLogLevel::Unknown(0xDEADBEEF), "Unknown error condition"),
        ];
        
        for (level, message) in error_cases {
            whisper_logging_trampoline_safe(level, std::borrow::Cow::Borrowed(message));
        }
    }
    
    #[test]
    fn test_whisper_logging_performance() {
        use std::time::Instant;
        
        // Test performance with repeated calls
        let start = Instant::now();
        
        for i in 0..1000 {
            whisper_logging_trampoline_safe(
                GGMLLogLevel::Info, 
                std::borrow::Cow::Owned(format!("Performance test message {}", i))
            );
        }
        
        let duration = start.elapsed();
        
        // Performance should be reasonable (less than 1 second for 1000 calls)
        assert!(duration.as_millis() < 1000);
    }
    
    #[test]
    fn test_whisper_logging_integration() {
        // Test integration with whisper logging installation
        install_whisper_logging_hook();
        
        // Test that we can call the safe function after installation
        whisper_logging_trampoline_safe(
            GGMLLogLevel::Info, 
            std::borrow::Cow::Borrowed("Integration test message")
        );
    }
    
    #[test]
    fn test_whisper_logging_thread_safety() {
        use std::thread;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        
        let success_flag = Arc::new(AtomicBool::new(true));
        
        // Test that logging is thread-safe
        let handles: Vec<_> = (0..5).map(|thread_id| {
            let flag = success_flag.clone();
            thread::spawn(move || {
                let result = std::panic::catch_unwind(|| {
                    for i in 0..20 {
                        whisper_logging_trampoline_safe(
                            GGMLLogLevel::Debug, 
                            std::borrow::Cow::Owned(format!("Thread {} message {}", thread_id, i))
                        );
                    }
                });
                
                if result.is_err() {
                    flag.store(false, Ordering::SeqCst);
                }
            })
        }).collect();
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert!(success_flag.load(Ordering::SeqCst), "Thread-safe logging failed");
    }
}
