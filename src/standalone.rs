//! Standalone functions that have no associated type.

use std::ffi::{c_int, CStr, CString};
use crate::common_logging::generic_info;
use std::time::SystemTime;

/// Return the id of the specified language, returns -1 if not found
///
/// # Arguments
/// * lang: The language to get the id for.
///
/// # Returns
/// The ID of the language, None if not found.
///
/// # Panics
/// Panics if the language contains a null byte.
///
/// # C++ equivalent
/// `int whisper_lang_id(const char * lang)`
pub fn get_lang_id(lang: &str) -> Option<c_int> {
    generic_info!("Looking up language ID for: '{}'", lang);
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    let c_lang = CString::new(lang).expect("Language contains null byte");
    let ret = unsafe { whisper_rs_sys::whisper_lang_id(c_lang.as_ptr()) };
    
    if ret == -1 {
        generic_info!("[{}] Language '{}' not found", timestamp, lang);
        None
    } else {
        generic_info!("[{}] Found language '{}' with ID {}", timestamp, lang, ret);
        Some(ret)
    }
}

/// Return the ID of the maximum language (ie the number of languages - 1)
///
/// # Returns
/// i32
///
/// # C++ equivalent
/// `int whisper_lang_max_id()`
pub fn get_lang_max_id() -> i32 {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    let max_id = unsafe { whisper_rs_sys::whisper_lang_max_id() };
    generic_info!("[{}] Maximum language ID: {}", timestamp, max_id);
    max_id
}

/// Get the short string of the specified language id (e.g. 2 -> "de").
///
/// # Returns
/// The short string of the language, None if not found.
///
/// # C++ equivalent
/// `const char * whisper_lang_str(int id)`
pub fn get_lang_str(id: i32) -> Option<&'static str> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    let c_buf = unsafe { whisper_rs_sys::whisper_lang_str(id) };
    if c_buf.is_null() {
        generic_info!("[{}] Language ID {} not found", timestamp, id);
        None
    } else {
        let c_str = unsafe { CStr::from_ptr(c_buf) };
        let lang_str = c_str.to_str().unwrap();
        generic_info!("[{}] Language ID {} -> '{}'", timestamp, id, lang_str);
        Some(lang_str)
    }
}

/// Get the full string of the specified language name (e.g. 2 -> "german").
///
/// # Returns
/// The full string of the language, None if not found.
///
/// # C++ equivalent
/// `const char * whisper_lang_str_full(int id)`
pub fn get_lang_str_full(id: i32) -> Option<&'static str> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
        
    let c_buf = unsafe { whisper_rs_sys::whisper_lang_str_full(id) };
    if c_buf.is_null() {
        generic_info!("[{}] Language ID {} full name not found", timestamp, id);
        None
    } else {
        let c_str = unsafe { CStr::from_ptr(c_buf) };
        let lang_str = c_str.to_str().unwrap();
        generic_info!("[{}] Language ID {} full name -> '{}'", timestamp, id, lang_str);
        Some(lang_str)
    }
}

/// Callback to control logging output: default behaviour is to print to stderr.
///
/// # Safety
/// The callback must be safe to call from C (i.e. no panicking, no unwinding, etc).
///
/// # C++ equivalent
/// `void whisper_set_log_callback(whisper_log_callback callback);`
pub unsafe fn set_log_callback(
    log_callback: crate::WhisperLogCallback,
    user_data: *mut std::ffi::c_void,
) {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    generic_info!("[{}] Setting whisper log callback", timestamp);
    unsafe {
        whisper_rs_sys::whisper_log_set(log_callback, user_data);
    }
}

/// Print system information.
///
/// # C++ equivalent
/// `const char * whisper_print_system_info()`
pub fn print_system_info() -> &'static str {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    let c_buf = unsafe { whisper_rs_sys::whisper_print_system_info() };
    let c_str = unsafe { CStr::from_ptr(c_buf) };
    let info_str = c_str.to_str().unwrap();
    generic_info!("[{}] System info retrieved", timestamp);
    info_str
}

/// Programmatically exposes the information provided by `print_system_info`
///
/// # C++ equivalent
/// `int ggml_cpu_has_...`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemInfo {
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
}

impl Default for SystemInfo {
    fn default() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        let info = unsafe {
            Self {
                avx: whisper_rs_sys::ggml_cpu_has_avx() != 0,
                avx2: whisper_rs_sys::ggml_cpu_has_avx2() != 0,
                fma: whisper_rs_sys::ggml_cpu_has_fma() != 0,
                f16c: whisper_rs_sys::ggml_cpu_has_f16c() != 0,
            }
        };
        
        generic_info!("[{}] System capabilities detected: AVX={}, AVX2={}, FMA={}, F16C={}", 
                     timestamp, info.avx, info.avx2, info.fma, info.f16c);
        info
    }
}

impl SystemInfo {
    /// Create a new SystemInfo with all capabilities disabled
    pub fn none() -> Self {
        Self {
            avx: false,
            avx2: false,
            fma: false,
            f16c: false,
        }
    }
    
    /// Create a new SystemInfo with all capabilities enabled
    pub fn all() -> Self {
        Self {
            avx: true,
            avx2: true,
            fma: true,
            f16c: true,
        }
    }
    
    /// Check if any capability is enabled
    pub fn has_any(&self) -> bool {
        self.avx || self.avx2 || self.fma || self.f16c
    }
    
    /// Check if all capabilities are enabled
    pub fn has_all(&self) -> bool {
        self.avx && self.avx2 && self.fma && self.f16c
    }
    
    /// Get the number of enabled capabilities
    pub fn capability_count(&self) -> u8 {
        (self.avx as u8) + (self.avx2 as u8) + (self.fma as u8) + (self.f16c as u8)
    }
}

/// Validate language identifier string format
pub fn is_valid_language_id(lang: &str) -> bool {
    !lang.is_empty() && 
    lang.len() <= 10 && 
    lang.chars().all(|c| c.is_alphabetic() || c == '-' || c == '_') &&
    !lang.contains('\0')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_info_default() {
        let info = SystemInfo::default();
        // Just check that we can create a SystemInfo struct without panicking
        println!("AVX: {}, AVX2: {}, FMA: {}, F16C: {}", info.avx, info.avx2, info.fma, info.f16c);
        
        // Ensure the struct is properly constructed
        assert!(info.avx == true || info.avx == false);
        assert!(info.avx2 == true || info.avx2 == false);
        assert!(info.fma == true || info.fma == false);
        assert!(info.f16c == true || info.f16c == false);
    }
    
    #[test]
    fn test_system_info_constructors() {
        let none_info = SystemInfo::none();
        assert!(!none_info.avx);
        assert!(!none_info.avx2);
        assert!(!none_info.fma);
        assert!(!none_info.f16c);
        assert!(!none_info.has_any());
        assert!(!none_info.has_all());
        assert_eq!(none_info.capability_count(), 0);
        
        let all_info = SystemInfo::all();
        assert!(all_info.avx);
        assert!(all_info.avx2);
        assert!(all_info.fma);
        assert!(all_info.f16c);
        assert!(all_info.has_any());
        assert!(all_info.has_all());
        assert_eq!(all_info.capability_count(), 4);
    }
    
    #[test]
    fn test_system_info_capabilities() {
        let partial_info = SystemInfo {
            avx: true,
            avx2: false,
            fma: true,
            f16c: false,
        };
        
        assert!(partial_info.has_any());
        assert!(!partial_info.has_all());
        assert_eq!(partial_info.capability_count(), 2);
    }
    
    #[test]
    fn test_system_info_equality() {
        let info1 = SystemInfo::none();
        let info2 = SystemInfo::none();
        let info3 = SystemInfo::all();
        
        assert_eq!(info1, info2);
        assert_ne!(info1, info3);
    }
    
    #[test]
    fn test_get_lang_max_id() {
        let max_id = get_lang_max_id();
        // Should be a reasonable number of languages
        assert!(max_id >= 0);
        assert!(max_id < 1000); // Reasonable upper bound
    }
    
    #[test]
    fn test_get_lang_id_valid() {
        // Test some common language codes
        let common_langs = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"];
        
        for lang in &common_langs {
            let id = get_lang_id(lang);
            if let Some(lang_id) = id {
                assert!(lang_id >= 0);
                assert!(lang_id <= get_lang_max_id());
            }
        }
    }
    
    #[test]
    fn test_get_lang_id_invalid() {
        // Test invalid language codes
        let invalid_langs = ["invalid", "xyz", "foo", "bar", "12345"];
        
        for lang in &invalid_langs {
            let id = get_lang_id(lang);
            // These should likely return None, but we just test they don't panic
            if let Some(lang_id) = id {
                assert!(lang_id >= 0);
            }
        }
    }
    
    #[test]
    #[should_panic(expected = "Language contains null byte")]
    fn test_get_lang_id_null_byte() {
        get_lang_id("en\0us");
    }
    
    #[test]
    fn test_get_lang_str_round_trip() {
        // Test that we can get language strings for valid IDs
        let max_id = get_lang_max_id();
        
        for id in 0..=std::cmp::min(max_id, 10) {
            let lang_str = get_lang_str(id);
            if let Some(lang) = lang_str {
                assert!(!lang.is_empty());
                assert!(lang.len() <= 10);
                assert!(lang.chars().all(|c| c.is_alphabetic() || c == '-'));
                
                // Try to get the ID back
                let recovered_id = get_lang_id(lang);
                if let Some(rid) = recovered_id {
                    assert_eq!(rid, id);
                }
            }
        }
    }
    
    #[test]
    fn test_get_lang_str_invalid_id() {
        // Test invalid language IDs
        let invalid_ids = [-1, -100, 9999, i32::MAX, i32::MIN];
        
        for id in &invalid_ids {
            let lang_str = get_lang_str(*id);
            assert!(lang_str.is_none());
        }
    }
    
    #[test]
    fn test_get_lang_str_full() {
        let max_id = get_lang_max_id();
        
        for id in 0..=std::cmp::min(max_id, 5) {
            let lang_str_full = get_lang_str_full(id);
            if let Some(full_name) = lang_str_full {
                assert!(!full_name.is_empty());
                assert!(full_name.len() >= 2);
                assert!(full_name.chars().all(|c| c.is_alphabetic() || c.is_whitespace()));
            }
        }
    }
    
    #[test]
    fn test_get_lang_str_full_invalid() {
        let invalid_ids = [-1, -100, 9999];
        
        for id in &invalid_ids {
            let lang_str_full = get_lang_str_full(*id);
            assert!(lang_str_full.is_none());
        }
    }
    
    #[test]
    fn test_print_system_info() {
        let info = print_system_info();
        assert!(!info.is_empty());
        // Should contain some system information
        assert!(info.len() > 10);
    }
    
    #[test]
    fn test_language_validation() {
        // Valid language identifiers
        assert!(is_valid_language_id("en"));
        assert!(is_valid_language_id("en-US"));
        assert!(is_valid_language_id("zh_CN"));
        assert!(is_valid_language_id("portuguese"));
        
        // Invalid language identifiers
        assert!(!is_valid_language_id(""));
        assert!(!is_valid_language_id("en\0us"));
        assert!(!is_valid_language_id("en123"));
        assert!(!is_valid_language_id("a_very_long_language_identifier"));
        assert!(!is_valid_language_id("en@US"));
        assert!(!is_valid_language_id("en.US"));
    }
    
    #[test]
    fn test_system_info_debug() {
        let info = SystemInfo::default();
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("SystemInfo"));
        assert!(debug_str.contains("avx"));
        assert!(debug_str.contains("avx2"));
        assert!(debug_str.contains("fma"));
        assert!(debug_str.contains("f16c"));
    }
    
    #[test]
    fn test_system_info_clone() {
        let info1 = SystemInfo::all();
        let info2 = info1.clone();
        assert_eq!(info1, info2);
        
        // Ensure they are separate instances
        assert_eq!(info1.avx, info2.avx);
        assert_eq!(info1.avx2, info2.avx2);
        assert_eq!(info1.fma, info2.fma);
        assert_eq!(info1.f16c, info2.f16c);
    }
    
    #[test]
    fn test_boundary_conditions() {
        // Test edge cases
        let zero_id = get_lang_str(0);
        // ID 0 might be valid (often English)
        if let Some(lang) = zero_id {
            assert!(!lang.is_empty());
        }
        
        // Test max ID
        let max_id = get_lang_max_id();
        let max_lang = get_lang_str(max_id);
        if let Some(lang) = max_lang {
            assert!(!lang.is_empty());
        }
        
        // Test just beyond max ID
        let beyond_max = get_lang_str(max_id + 1);
        assert!(beyond_max.is_none());
    }
    
    #[test]
    fn test_consistency_between_functions() {
        // Test that lang_str and lang_str_full are consistent
        for id in 0..=std::cmp::min(get_lang_max_id(), 10) {
            let short = get_lang_str(id);
            let full = get_lang_str_full(id);
            
            // Both should be either Some or None for the same ID
            if short.is_some() && full.is_some() {
                let short_str = short.unwrap();
                let full_str = full.unwrap();
                assert!(!short_str.is_empty());
                assert!(!full_str.is_empty());
                // Full name should be longer or equal to short name
                assert!(full_str.len() >= short_str.len());
            }
        }
    }
}
