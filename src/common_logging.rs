/// Macro for logging error messages with timestamp
/// 
/// This macro is for internal use within the whisper-rs crate.
macro_rules! generic_error {
    ($($expr:tt)*) => {
        #[cfg(feature = "log_backend")]
        log::error!($($expr)*);
        #[cfg(feature = "tracing_backend")]
        tracing::error!($($expr)*);
    };
}

/// Macro for logging warning messages with timestamp
/// 
/// This macro is for internal use within the whisper-rs crate.
macro_rules! generic_warn {
    ($($expr:tt)*) => {
        #[cfg(feature = "log_backend")]
        log::warn!($($expr)*);
        #[cfg(feature = "tracing_backend")]
        tracing::warn!($($expr)*);
    }
}

/// Macro for logging info messages with timestamp
/// 
/// This macro is for internal use within the whisper-rs crate.
macro_rules! generic_info {
    ($($expr:tt)*) => {
        #[cfg(feature = "log_backend")]
        log::info!($($expr)*);
        #[cfg(feature = "tracing_backend")]
        tracing::info!($($expr)*);
    }
}

/// Macro for logging debug messages with timestamp
/// 
/// This macro is for internal use within the whisper-rs crate.
macro_rules! generic_debug {
    ($($expr:tt)*) => {
        #[cfg(feature = "log_backend")]
        log::debug!($($expr)*);
        #[cfg(feature = "tracing_backend")]
        tracing::debug!($($expr)*);
    }
}

/// Macro for logging trace messages with timestamp
/// 
/// This macro is for internal use within the whisper-rs crate.
macro_rules! generic_trace {
    ($($expr:tt)*) => {
        #[cfg(feature = "log_backend")]
        log::trace!($($expr)*);
        #[cfg(feature = "tracing_backend")]
        tracing::trace!($($expr)*);
    }
}

use whisper_rs_sys::ggml_log_level;
pub(crate) use {generic_debug, generic_error, generic_info, generic_trace, generic_warn};

// Unsigned integer type on most platforms is 32 bit, niche platforms that whisper.cpp
// likely doesn't even support would use 16 bit and would still fit
#[cfg_attr(any(not(windows), target_env = "gnu"), repr(u32))]
// Of course Windows thinks it's a special little shit and
// picks a signed integer for an unsigned type
#[cfg_attr(all(windows, not(target_env = "gnu")), repr(i32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GGMLLogLevel {
    None = whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_NONE,
    Info = whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_INFO,
    Warn = whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_WARN,
    Error = whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_ERROR,
    Debug = whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_DEBUG,
    Cont = whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_CONT,
    Unknown(ggml_log_level),
}

impl GGMLLogLevel {
    /// Get the severity level as a string
    /// 
    /// # Examples
    /// ```
    /// # use whisper_rs::GGMLLogLevel;
    /// assert_eq!(GGMLLogLevel::Error.to_string_level(), "ERROR");
    /// assert_eq!(GGMLLogLevel::Debug.to_string_level(), "DEBUG");
    /// ```
    pub fn to_string_level(&self) -> &'static str {
        match self {
            GGMLLogLevel::None => "NONE",
            GGMLLogLevel::Info => "INFO",
            GGMLLogLevel::Warn => "WARN",
            GGMLLogLevel::Error => "ERROR",
            GGMLLogLevel::Debug => "DEBUG",
            GGMLLogLevel::Cont => "CONT",
            GGMLLogLevel::Unknown(_) => "UNKNOWN",
        }
    }
    
    /// Check if this log level should be displayed given a minimum level
    /// 
    /// # Arguments
    /// * `min_level` - The minimum log level to display
    /// 
    /// # Returns
    /// `true` if this level should be logged, `false` otherwise
    pub fn should_log(&self, min_level: GGMLLogLevel) -> bool {
        let self_priority = self.priority();
        let min_priority = min_level.priority();
        self_priority >= min_priority
    }
    
    /// Get numeric priority for comparison (higher = more important)
    fn priority(&self) -> i32 {
        match self {
            GGMLLogLevel::None => 0,
            GGMLLogLevel::Debug => 1,
            GGMLLogLevel::Info => 2,
            GGMLLogLevel::Warn => 3,
            GGMLLogLevel::Error => 4,
            GGMLLogLevel::Cont => 5,
            GGMLLogLevel::Unknown(_) => -1,
        }
    }
    
    /// Create from raw value with bounds checking
    /// 
    /// # Arguments
    /// * `value` - Raw log level value
    /// 
    /// # Returns
    /// Valid GGMLLogLevel or Unknown variant for invalid values
    pub fn from_raw(value: ggml_log_level) -> Self {
        Self::from(value)
    }
}

impl From<ggml_log_level> for GGMLLogLevel {
    fn from(level: ggml_log_level) -> Self {
        match level {
            whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_NONE => GGMLLogLevel::None,
            whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_INFO => GGMLLogLevel::Info,
            whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_WARN => GGMLLogLevel::Warn,
            whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_ERROR => GGMLLogLevel::Error,
            whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_DEBUG => GGMLLogLevel::Debug,
            whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_CONT => GGMLLogLevel::Cont,
            other => GGMLLogLevel::Unknown(other),
        }
    }
}

impl From<GGMLLogLevel> for ggml_log_level {
    fn from(level: GGMLLogLevel) -> Self {
        match level {
            GGMLLogLevel::None => whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_NONE,
            GGMLLogLevel::Info => whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_INFO,
            GGMLLogLevel::Warn => whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_WARN,
            GGMLLogLevel::Error => whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_ERROR,
            GGMLLogLevel::Debug => whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_DEBUG,
            GGMLLogLevel::Cont => whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_CONT,
            GGMLLogLevel::Unknown(value) => value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ggml_log_level_string_conversion() {
        assert_eq!(GGMLLogLevel::None.to_string_level(), "NONE");
        assert_eq!(GGMLLogLevel::Info.to_string_level(), "INFO");
        assert_eq!(GGMLLogLevel::Warn.to_string_level(), "WARN");
        assert_eq!(GGMLLogLevel::Error.to_string_level(), "ERROR");
        assert_eq!(GGMLLogLevel::Debug.to_string_level(), "DEBUG");
        assert_eq!(GGMLLogLevel::Cont.to_string_level(), "CONT");
        assert_eq!(GGMLLogLevel::Unknown(999).to_string_level(), "UNKNOWN");
    }
    
    #[test]
    fn test_ggml_log_level_priority() {
        // Test priority ordering
        assert!(GGMLLogLevel::Error.priority() > GGMLLogLevel::Warn.priority());
        assert!(GGMLLogLevel::Warn.priority() > GGMLLogLevel::Info.priority());
        assert!(GGMLLogLevel::Info.priority() > GGMLLogLevel::Debug.priority());
        assert!(GGMLLogLevel::Debug.priority() > GGMLLogLevel::None.priority());
        assert!(GGMLLogLevel::Cont.priority() > GGMLLogLevel::Error.priority());
        assert_eq!(GGMLLogLevel::Unknown(123).priority(), -1);
    }
    
    #[test]
    fn test_should_log_functionality() {
        // Error should log at warn level
        assert!(GGMLLogLevel::Error.should_log(GGMLLogLevel::Warn));
        
        // Debug should not log at info level
        assert!(!GGMLLogLevel::Debug.should_log(GGMLLogLevel::Info));
        
        // Same levels should log
        assert!(GGMLLogLevel::Info.should_log(GGMLLogLevel::Info));
        
        // Unknown levels should not log
        assert!(!GGMLLogLevel::Unknown(123).should_log(GGMLLogLevel::Debug));
    }
    
    #[test]
    fn test_log_level_conversions() {
        // Test round-trip conversion for valid levels
        let levels = vec![
            GGMLLogLevel::None,
            GGMLLogLevel::Info,
            GGMLLogLevel::Warn,
            GGMLLogLevel::Error,
            GGMLLogLevel::Debug,
            GGMLLogLevel::Cont,
        ];
        
        for level in levels {
            let raw: ggml_log_level = level.into();
            let converted_back = GGMLLogLevel::from(raw);
            assert_eq!(level, converted_back);
        }
    }
    
    #[test]
    fn test_unknown_log_level_handling() {
        let unknown_value = 999;
        let level = GGMLLogLevel::from(unknown_value);
        
        match level {
            GGMLLogLevel::Unknown(value) => assert_eq!(value, unknown_value),
            _ => panic!("Expected Unknown variant"),
        }
        
        // Test round-trip for unknown
        let back_to_raw: ggml_log_level = level.into();
        assert_eq!(back_to_raw, unknown_value);
    }
    
    #[test]
    fn test_from_raw_method() {
        let level = GGMLLogLevel::from_raw(whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_ERROR);
        assert_eq!(level, GGMLLogLevel::Error);
        
        let unknown = GGMLLogLevel::from_raw(9999);
        assert!(matches!(unknown, GGMLLogLevel::Unknown(9999)));
    }
    
    #[test]
    fn test_log_level_traits() {
        let level1 = GGMLLogLevel::Error;
        let level2 = GGMLLogLevel::Error;
        let level3 = GGMLLogLevel::Warn;
        
        // Test PartialEq
        assert_eq!(level1, level2);
        assert_ne!(level1, level3);
        
        // Test Clone and Copy
        let cloned = level1.clone();
        let copied = level1;
        assert_eq!(level1, cloned);
        assert_eq!(level1, copied);
        
        // Test Debug
        let debug_str = format!("{:?}", level1);
        assert!(debug_str.contains("Error"));
    }
    
    #[test]
    fn test_logging_macros_do_not_panic() {
        // These should not panic even if logging is not configured
        generic_error!("Test error: {}", "message");
        generic_warn!("Test warning: {}", 42);
        generic_info!("Test info");
        generic_debug!("Test debug: {} {}", "arg1", "arg2");
        generic_trace!("Test trace");
    }
    
    #[test]
    fn test_boundary_conditions() {
        // Test with maximum and minimum values
        let max_level = GGMLLogLevel::from(u32::MAX);
        let min_level = GGMLLogLevel::from(0u32);
        
        assert!(matches!(max_level, GGMLLogLevel::Unknown(_)));
        assert_eq!(min_level, GGMLLogLevel::None);
        
        // Ensure they don't panic in string conversion
        assert_eq!(max_level.to_string_level(), "UNKNOWN");
        assert_eq!(min_level.to_string_level(), "NONE");
    }
}
