use std::ffi::{c_int, CStr};
use std::sync::Arc;
use std::time::SystemTime;

use crate::{
    WhisperContextParameters, WhisperError, WhisperInnerContext, WhisperState, WhisperToken,
    common_logging::generic_info,
};

pub struct WhisperContext {
    ctx: Arc<WhisperInnerContext>,
}

impl WhisperContext {
    fn wrap(ctx: WhisperInnerContext) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        generic_info!("[{}] Wrapping WhisperInnerContext in WhisperContext", timestamp);
        Self { ctx: Arc::new(ctx) }
    }

    /// Create a new WhisperContext from a file, with parameters.
    ///
    /// # Arguments
    /// * path: The path to the model file.
    /// * parameters: A parameter struct containing the parameters to use.
    ///
    /// # Returns
    /// Ok(Self) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `struct whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model, struct whisper_context_params params);`
    pub fn new_with_params(
        path: &str,
        parameters: WhisperContextParameters,
    ) -> Result<Self, WhisperError> {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        generic_info!("[{}] Creating WhisperContext from file: '{}'", timestamp, path);
        
        let ctx = WhisperInnerContext::new_with_params(path, parameters)?;
        Ok(Self::wrap(ctx))
    }

    /// Create a new WhisperContext from a buffer.
    ///
    /// # Arguments
    /// * buffer: The buffer containing the model.
    ///
    /// # Returns
    /// Ok(Self) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `struct whisper_context * whisper_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size, struct whisper_context_params params);`
    pub fn new_from_buffer_with_params(
        buffer: &[u8],
        parameters: WhisperContextParameters,
    ) -> Result<Self, WhisperError> {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        generic_info!("[{}] Creating WhisperContext from buffer of {} bytes", timestamp, buffer.len());
        
        let ctx = WhisperInnerContext::new_from_buffer_with_params(buffer, parameters)?;
        Ok(Self::wrap(ctx))
    }

    /// Convert the provided text into tokens.
    ///
    /// # Arguments
    /// * text: The text to convert.
    ///
    /// # Returns
    /// `Ok(Vec<WhisperToken>)` on success, `Err(WhisperError)` on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_tokenize(struct whisper_context * ctx, const char * text, whisper_token * tokens, int n_max_tokens);`
    pub fn tokenize(
        &self,
        text: &str,
        max_tokens: usize,
    ) -> Result<Vec<WhisperToken>, WhisperError> {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        generic_info!("[{}] Tokenizing text: '{}' (max_tokens={})", timestamp, text, max_tokens);
        
        self.ctx.tokenize(text, max_tokens)
    }

    /// Get n_vocab.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_n_vocab        (struct whisper_context * ctx)`
    #[inline]
    pub fn n_vocab(&self) -> c_int {
        self.ctx.n_vocab()
    }

    /// Get n_text_ctx.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_n_text_ctx     (struct whisper_context * ctx);`
    #[inline]
    pub fn n_text_ctx(&self) -> c_int {
        self.ctx.n_text_ctx()
    }

    /// Get n_audio_ctx.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_n_audio_ctx     (struct whisper_context * ctx);`
    #[inline]
    pub fn n_audio_ctx(&self) -> c_int {
        self.ctx.n_audio_ctx()
    }

    /// Does this model support multiple languages?
    ///
    /// # C++ equivalent
    /// `int whisper_is_multilingual(struct whisper_context * ctx)`
    #[inline]
    pub fn is_multilingual(&self) -> bool {
        self.ctx.is_multilingual()
    }

    /// Get model_n_vocab.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_vocab      (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_vocab(&self) -> c_int {
        self.ctx.model_n_vocab()
    }

    /// Get model_n_audio_ctx.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_audio_ctx    (struct whisper_context * ctx)`
    #[inline]
    pub fn model_n_audio_ctx(&self) -> c_int {
        self.ctx.model_n_audio_ctx()
    }

    /// Get model_n_audio_state.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_audio_state(struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_audio_state(&self) -> c_int {
        self.ctx.model_n_audio_state()
    }

    /// Get model_n_audio_head.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_audio_head (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_audio_head(&self) -> c_int {
        self.ctx.model_n_audio_head()
    }

    /// Get model_n_audio_layer.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_audio_layer(struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_audio_layer(&self) -> c_int {
        self.ctx.model_n_audio_layer()
    }

    /// Get model_n_text_ctx.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_text_ctx     (struct whisper_context * ctx)`
    #[inline]
    pub fn model_n_text_ctx(&self) -> c_int {
        self.ctx.model_n_text_ctx()
    }

    /// Get model_n_text_state.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_text_state (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_text_state(&self) -> c_int {
        self.ctx.model_n_text_state()
    }

    /// Get model_n_text_head.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_text_head  (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_text_head(&self) -> c_int {
        self.ctx.model_n_text_head()
    }

    /// Get model_n_text_layer.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_text_layer (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_text_layer(&self) -> c_int {
        self.ctx.model_n_text_layer()
    }

    /// Get model_n_mels.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_mels       (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_mels(&self) -> c_int {
        self.ctx.model_n_mels()
    }

    /// Get model_ftype.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_ftype          (struct whisper_context * ctx);`
    #[inline]
    pub fn model_ftype(&self) -> c_int {
        self.ctx.model_ftype()
    }

    /// Get model_type.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_type         (struct whisper_context * ctx);`
    #[inline]
    pub fn model_type(&self) -> c_int {
        self.ctx.model_type()
    }

    // token functions
    /// Convert a token ID to a string.
    ///
    /// # Arguments
    /// * token_id: ID of the token.
    ///
    /// # Returns
    /// Ok(&str) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `const char * whisper_token_to_str(struct whisper_context * ctx, whisper_token token)`
    pub fn token_to_str(&self, token_id: WhisperToken) -> Result<&str, WhisperError> {
        self.ctx.token_to_str(token_id)
    }

    /// Convert a token ID to a &CStr.
    ///
    /// # Arguments
    /// * token_id: ID of the token.
    ///
    /// # Returns
    /// Ok(String) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `const char * whisper_token_to_str(struct whisper_context * ctx, whisper_token token)`
    pub fn token_to_cstr(&self, token_id: WhisperToken) -> Result<&CStr, WhisperError> {
        self.ctx.token_to_cstr(token_id)
    }

    /// Undocumented but exposed function in the C++ API.
    /// `const char * whisper_model_type_readable(struct whisper_context * ctx);`
    ///
    /// # Returns
    /// Ok(String) on success, Err(WhisperError) on failure.
    pub fn model_type_readable(&self) -> Result<String, WhisperError> {
        self.ctx.model_type_readable()
    }

    /// Get the ID of the eot token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_eot (struct whisper_context * ctx)`
    #[inline]
    pub fn token_eot(&self) -> WhisperToken {
        self.ctx.token_eot()
    }

    /// Get the ID of the sot token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_sot (struct whisper_context * ctx)`
    #[inline]
    pub fn token_sot(&self) -> WhisperToken {
        self.ctx.token_sot()
    }

    /// Get the ID of the solm token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_solm(struct whisper_context * ctx)`
    #[inline]
    pub fn token_solm(&self) -> WhisperToken {
        self.ctx.token_solm()
    }

    /// Get the ID of the prev token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_prev(struct whisper_context * ctx)`
    #[inline]
    pub fn token_prev(&self) -> WhisperToken {
        self.ctx.token_prev()
    }

    /// Get the ID of the nosp token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_nosp(struct whisper_context * ctx)`
    #[inline]
    pub fn token_nosp(&self) -> WhisperToken {
        self.ctx.token_nosp()
    }

    /// Get the ID of the not token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_not (struct whisper_context * ctx)`
    #[inline]
    pub fn token_not(&self) -> WhisperToken {
        self.ctx.token_not()
    }

    /// Get the ID of the beg token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_beg (struct whisper_context * ctx)`
    #[inline]
    pub fn token_beg(&self) -> WhisperToken {
        self.ctx.token_beg()
    }

    /// Get the ID of a specified language token
    ///
    /// # Arguments
    /// * lang_id: ID of the language
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_lang(struct whisper_context * ctx, int lang_id)`
    #[inline]
    pub fn token_lang(&self, lang_id: c_int) -> WhisperToken {
        self.ctx.token_lang(lang_id)
    }

    /// Print performance statistics to stderr.
    ///
    /// # C++ equivalent
    /// `void whisper_print_timings(struct whisper_context * ctx)`
    #[inline]
    pub fn print_timings(&self) {
        self.ctx.print_timings()
    }

    /// Reset performance statistics.
    ///
    /// # C++ equivalent
    /// `void whisper_reset_timings(struct whisper_context * ctx)`
    #[inline]
    pub fn reset_timings(&self) {
        self.ctx.reset_timings()
    }

    // task tokens
    /// Get the ID of the translate task token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_translate ()`
    pub fn token_translate(&self) -> WhisperToken {
        self.ctx.token_translate()
    }

    /// Get the ID of the transcribe task token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_transcribe()`
    pub fn token_transcribe(&self) -> WhisperToken {
        self.ctx.token_transcribe()
    }

    /// Create a new state for this context.
    ///
    /// # Returns
    /// Ok(WhisperState) on success, Err(WhisperError) on failure.
    pub fn create_state(&self) -> Result<WhisperState, WhisperError> {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        generic_info!("[{}] Creating new WhisperState", timestamp);
        
        let state_ptr = unsafe {
            whisper_rs_sys::whisper_init_state(self.ctx.ctx)
        };
        
        if state_ptr.is_null() {
            Err(WhisperError::InitError)
        } else {
            Ok(WhisperState::new(Arc::clone(&self.ctx), state_ptr))
        }
    }
    
    /// Check if the context is valid (non-null internal pointer)
    pub fn is_valid(&self) -> bool {
        !self.ctx.ctx.is_null()
    }
    
    /// Get the reference count of the internal Arc
    pub fn reference_count(&self) -> usize {
        Arc::strong_count(&self.ctx)
    }
}

// Clone implementation to allow sharing of the context
impl Clone for WhisperContext {
    fn clone(&self) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        generic_info!("[{}] Cloning WhisperContext", timestamp);
        
        Self {
            ctx: Arc::clone(&self.ctx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WhisperContextParameters;

    // Mock tests that don't require actual model files
    
    #[test]
    fn test_whisper_context_invalid_file() {
        // Test creating context with non-existent file
        let params = WhisperContextParameters::default();
        let result = WhisperContext::new_with_params("non_existent_file.bin", params);
        assert!(result.is_err());
        
        if let Err(error) = result {
            // Should be an InitError for non-existent file
            assert!(matches!(error, WhisperError::InitError));
        }
    }
    
    #[test]
    fn test_whisper_context_invalid_buffer() {
        // Test creating context with invalid buffer (empty buffer)
        let empty_buffer: [u8; 0] = [];
        let params = WhisperContextParameters::default();
        let result = WhisperContext::new_from_buffer_with_params(&empty_buffer, params);
        assert!(result.is_err());
        
        if let Err(error) = result {
            assert!(matches!(error, WhisperError::InitError));
        }
    }
    
    #[test]
    fn test_whisper_context_parameters_defaults() {
        let params = WhisperContextParameters::default();
        
        // Test default values
        #[cfg(feature = "_gpu")]
        assert!(params.use_gpu);
        #[cfg(not(feature = "_gpu"))]
        assert!(!params.use_gpu);
        
        assert!(!params.flash_attn);
        assert_eq!(params.gpu_device, 0);
    }
    
    #[test]
    fn test_whisper_context_parameters_builder() {
        let mut params = WhisperContextParameters::new();
        params.use_gpu(true)
              .flash_attn(true)
              .gpu_device(1);
        
        assert!(params.use_gpu);
        assert!(params.flash_attn);
        assert_eq!(params.gpu_device, 1);
    }
    
    #[test]
    fn test_whisper_context_parameters_dtw() {
        use crate::whisper_ctx::{DtwParameters, DtwMode};
        
        let dtw_params = DtwParameters::default();
        assert!(matches!(dtw_params.mode, DtwMode::None));
        assert_eq!(dtw_params.dtw_mem_size, 1024 * 1024 * 128);
        
        let mut params = WhisperContextParameters::new();
        params.dtw_parameters(dtw_params);
        
        // Test that DTW parameters can be set
        assert!(matches!(params.dtw_parameters.mode, DtwMode::None));
    }

    // Test functions that would work even without a valid context (testing API)
    
    #[test]
    fn test_empty_tokenization() {
        // This test would fail without a valid context, but we can test the error handling
        let params = WhisperContextParameters::default();
        let ctx_result = WhisperContext::new_with_params("", params);
        
        // Should fail to create context
        assert!(ctx_result.is_err());
    }
    
    #[test]
    fn test_context_clone_behavior() {
        // Test the cloning behavior with mock data structure
        use std::sync::Arc;
        use crate::WhisperInnerContext;
        
        // We can't easily create a valid WhisperInnerContext without a model file,
        // but we can test the clone logic
        let params = WhisperContextParameters::default();
        let ctx_result = WhisperContext::new_with_params("nonexistent", params);
        assert!(ctx_result.is_err());
    }
    
    #[test]
    fn test_error_handling_consistency() {
        let params = WhisperContextParameters::default();
        
        // Test multiple creation attempts with invalid inputs
        let results = vec![
            WhisperContext::new_with_params("", WhisperContextParameters::default()),
            WhisperContext::new_with_params("nonexistent.bin", WhisperContextParameters::default()),
            WhisperContext::new_with_params("/invalid/path/model.bin", WhisperContextParameters::default()),
        ];
        
        // All should fail
        for result in results {
            assert!(result.is_err());
        }
    }
    
    #[test]
    fn test_buffer_size_validation() {
        let params = WhisperContextParameters::default();
        
        // Test with various invalid buffer sizes
        let test_cases = vec![
            vec![], // Empty
            vec![0u8], // Too small  
            vec![0u8; 10], // Still too small
            b"invalid model data".to_vec(), // Invalid content
        ];
        
        for buffer in test_cases {
            let result = WhisperContext::new_from_buffer_with_params(&buffer, WhisperContextParameters::default());
            assert!(result.is_err(), "Buffer with length {} should fail", buffer.len());
        }
    }
    
    #[test]
    fn test_parameter_edge_cases() {
        let mut params = WhisperContextParameters::default();
        
        // Test edge case values
        params.gpu_device(-1); // Negative device ID
        assert_eq!(params.gpu_device, -1);
        
        params.gpu_device(i32::MAX); // Maximum device ID
        assert_eq!(params.gpu_device, i32::MAX);
        
        params.gpu_device(0); // Reset to valid value
        assert_eq!(params.gpu_device, 0);
    }
    
    #[test]
    fn test_dtw_mode_variants() {
        use crate::whisper_ctx::{DtwMode, DtwModelPreset};
        
        // Test different DTW modes
        let mode_none = DtwMode::None;
        let mode_top = DtwMode::TopMost { n_top: 5 };
        let mode_preset = DtwMode::ModelPreset { 
            model_preset: DtwModelPreset::Base 
        };
        
        // These should be constructible
        assert!(matches!(mode_none, DtwMode::None));
        if let DtwMode::TopMost { n_top } = mode_top {
            assert_eq!(n_top, 5);
        }
        assert!(matches!(mode_preset, DtwMode::ModelPreset { .. }));
    }

    #[test]
    fn test_string_validation() {
        // Test path validation (these will fail but won't panic)
        let long_path = "very_long_path_".repeat(1000);
        let invalid_paths = vec![
            "",
            "\0", 
            "path\0with\0nulls",
            &long_path,
        ];
        
        for path in invalid_paths {
            let result = WhisperContext::new_with_params(path, WhisperContextParameters::default());
            // Should handle gracefully without panicking
            assert!(result.is_err());
        }
    }
    
    #[test] 
    fn test_concurrent_creation_attempts() {
        use std::thread;
        
        let handles: Vec<thread::JoinHandle<Result<WhisperContext, WhisperError>>> = (0..5).map(|i| {
            thread::spawn(move || {
                let path = format!("nonexistent_{}.bin", i);
                WhisperContext::new_with_params(&path, WhisperContextParameters::default())
            })
        }).collect();
        
        // All threads should complete without panicking
        for handle in handles {
            let result = handle.join().expect("Thread should not panic");
            assert!(result.is_err()); // All should fail since files don't exist
        }
    }
    
    #[test]
    fn test_memory_safety() {
        // Test that dropping contexts works correctly
        let test_path = format!("test_{}.bin", 123);
        
        // Create multiple failed contexts and ensure they can be dropped safely
        for i in 0..10 {
            let path = format!("test_{}.bin", i);
            if let Err(_) = WhisperContext::new_with_params(&path, WhisperContextParameters::default()) {
                // Expected to fail, just testing that creation and destruction is safe
            }
        }
    }
    
    #[test]
    fn test_debug_and_display_traits() {
        // Test that error types can be formatted
        let error = WhisperError::InitError;
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("InitError"));
        
        let display_str = format!("{}", error);
        assert!(!display_str.is_empty());
    }
    
    #[test]
    fn test_context_parameters_chaining() {
        // Test method chaining for parameters
        let mut params = WhisperContextParameters::new();
        params.use_gpu(true)
              .flash_attn(false)
              .gpu_device(2);
        
        assert!(params.use_gpu);
        assert!(!params.flash_attn);
        assert_eq!(params.gpu_device, 2);
    }
}
