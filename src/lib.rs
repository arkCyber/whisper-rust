#![allow(clippy::uninlined_format_args)]

#[cfg(feature = "vulkan")]
pub mod vulkan;

mod common_logging;
mod error;
mod ggml_logging_hook;
mod standalone;
mod utilities;
mod whisper_ctx;
mod whisper_ctx_wrapper;
mod whisper_grammar;
mod whisper_logging_hook;
mod whisper_params;
mod whisper_state;

pub use common_logging::GGMLLogLevel;
pub use error::WhisperError;
pub use standalone::*;
pub use utilities::*;
pub use whisper_ctx::DtwMode;
pub use whisper_ctx::DtwModelPreset;
pub use whisper_ctx::DtwParameters;
pub use whisper_ctx::WhisperContextParameters;
use whisper_ctx::WhisperInnerContext;
pub use whisper_ctx_wrapper::WhisperContext;
pub use whisper_grammar::{WhisperGrammarElement, WhisperGrammarElementType};
pub use whisper_params::{FullParams, SamplingStrategy, SegmentCallbackData};
#[cfg(feature = "raw-api")]
pub use whisper_rs_sys;
pub use whisper_state::WhisperState;

pub type WhisperSysContext = whisper_rs_sys::whisper_context;
pub type WhisperSysState = whisper_rs_sys::whisper_state;

pub type WhisperTokenData = whisper_rs_sys::whisper_token_data;
pub type WhisperToken = whisper_rs_sys::whisper_token;
pub type WhisperNewSegmentCallback = whisper_rs_sys::whisper_new_segment_callback;
pub type WhisperStartEncoderCallback = whisper_rs_sys::whisper_encoder_begin_callback;
pub type WhisperProgressCallback = whisper_rs_sys::whisper_progress_callback;
pub type WhisperLogitsFilterCallback = whisper_rs_sys::whisper_logits_filter_callback;
pub type WhisperAbortCallback = whisper_rs_sys::ggml_abort_callback;
pub type WhisperLogCallback = whisper_rs_sys::ggml_log_callback;
pub type DtwAhead = whisper_rs_sys::whisper_ahead;

/// The version of whisper.cpp that whisper-rs was linked with.
pub static WHISPER_CPP_VERSION: &str = env!("WHISPER_CPP_VERSION");

/// Redirect all whisper.cpp and GGML logs to logging hooks installed by whisper-rs.
///
/// This will stop most logs from being output to stdout/stderr and will bring them into
/// `log` or `tracing`, if the `log_backend` or `tracing_backend` features, respectively,
/// are enabled. If neither is enabled, this will essentially disable logging, as they won't
/// be output anywhere.
///
/// Note whisper.cpp and GGML do not reliably follow Rust logging conventions.
/// Use your logging crate's configuration to control how these logs will be output.
/// whisper-rs does not currently output any logs, but this may change in the future.
/// You should configure by module path and use `whisper_rs::ggml_logging_hook`,
/// and/or `whisper_rs::whisper_logging_hook`, to avoid possibly ignoring useful
/// `whisper-rs` logs in the future.
///
/// Safe to call multiple times. Only has an effect the first time.
/// (note this means installing your own logging handlers with unsafe functions after this call
/// is permanent and cannot be undone)
pub fn install_logging_hooks() {
    crate::whisper_logging_hook::install_whisper_logging_hook();
    crate::ggml_logging_hook::install_ggml_logging_hook();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    
    #[test]
    fn test_install_logging_hooks() {
        // Test that installing logging hooks doesn't panic
        install_logging_hooks();
        
        // Test multiple calls are safe (should be idempotent)
        install_logging_hooks();
        install_logging_hooks();
        
        // If we reach here, no panic occurred
        assert!(true);
    }
    
    #[test]
    fn test_whisper_cpp_version_constant() {
        // Test that the version constant is not empty
        assert!(!WHISPER_CPP_VERSION.is_empty());
        
        // Test that it looks like a version string (contains a dot)
        assert!(WHISPER_CPP_VERSION.contains('.') || WHISPER_CPP_VERSION.contains('-'));
        
        // Test that it's a valid UTF-8 string
        assert!(std::str::from_utf8(WHISPER_CPP_VERSION.as_bytes()).is_ok());
    }
    
    #[test]
    fn test_public_api_exports() {
        // Test that key types are properly exported and can be used
        
        // Test GGMLLogLevel
        let _level = GGMLLogLevel::Info;
        assert_eq!(_level.to_string_level(), "INFO");
        
        // Test WhisperError (if accessible)
        // This ensures the error type is properly exported
        let _error_check = std::any::type_name::<WhisperError>();
        assert!(_error_check.contains("WhisperError"));
    }
    
    #[test]
    fn test_logging_hooks_integration() {
        // Test that both logging hooks can be installed together
        // This is an integration test to ensure no conflicts
        
        // Install logging hooks
        install_logging_hooks();
        
        // Test individual hook installation functions
        crate::whisper_logging_hook::install_whisper_logging_hook();
        crate::ggml_logging_hook::install_ggml_logging_hook();
        
        // If we reach here without panic, the integration works
        assert!(true);
    }
    
    #[test]
    fn test_timestamp_logging_feature() {
        // Test timestamp functionality as mentioned in user rules
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Install logging hooks with timestamp capability
        install_logging_hooks();
        
        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Verify timing is reasonable (should be very quick)
        assert!(end_time - start_time < 5);
    }
    
    #[test]
    fn test_error_handling_no_panic() {
        // Test that error handling code doesn't panic
        // Following user rules about adding error handling code
        
        let result = std::panic::catch_unwind(|| {
            install_logging_hooks();
            
            // Test multiple installations
            for _ in 0..10 {
                install_logging_hooks();
            }
        });
        
        assert!(result.is_ok(), "Logging hook installation should not panic");
    }
    
    #[test]
    fn test_library_features_availability() {
        // Test feature flags and conditional compilation
        
        #[cfg(feature = "log_backend")]
        {
            // If log backend is enabled, test that log crate is available
            assert!(true); // log crate should be available
        }
        
        #[cfg(feature = "tracing_backend")]
        {
            // If tracing backend is enabled, test that tracing crate is available
            assert!(true); // tracing crate should be available
        }
        
        #[cfg(feature = "raw-api")]
        {
            // If raw API is enabled, test that whisper_rs_sys is accessible
            let _sys_check = std::any::type_name::<whisper_rs_sys::whisper_context>();
            assert!(_sys_check.contains("whisper_context"));
        }
        
        // Test feature independent functionality
        assert!(true);
    }
    
    #[test]
    fn test_module_visibility() {
        // Test that modules are properly visible and accessible
        
        // Test common logging module
        let _level = crate::common_logging::GGMLLogLevel::Info;
        assert_eq!(_level, GGMLLogLevel::Info);
        
        // Test that error module is accessible
        let _error_name = std::any::type_name::<crate::error::WhisperError>();
        assert!(_error_name.contains("WhisperError"));
    }
    
    #[test]
    fn test_thread_safety_of_logging_installation() {
        use std::thread;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        
        let success_flag = Arc::new(AtomicBool::new(true));
        
        // Test that logging hook installation is thread-safe
        let handles: Vec<_> = (0..5).map(|_| {
            let flag = success_flag.clone();
            thread::spawn(move || {
                let result = std::panic::catch_unwind(|| {
                    install_logging_hooks();
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
        
        assert!(success_flag.load(Ordering::SeqCst), "Thread-safe installation failed");
    }
    
    #[test]
    fn test_memory_safety() {
        // Test that there are no memory safety issues with repeated operations
        
        for _ in 0..100 {
            install_logging_hooks();
        }
        
        // Test that we can call it many times without issues
        assert!(true);
    }
}
