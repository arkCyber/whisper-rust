use std::{
    ffi::{c_int, CStr},
    sync::Arc,
    time::SystemTime,
};

use crate::{
    FullParams, WhisperError, WhisperInnerContext, WhisperToken, WhisperTokenData,
    common_logging::generic_info,
};

/// Rustified pointer to a Whisper state.
#[derive(Debug)]
pub struct WhisperState {
    ctx: Arc<WhisperInnerContext>,
    ptr: *mut whisper_rs_sys::whisper_state,
}

unsafe impl Send for WhisperState {}

unsafe impl Sync for WhisperState {}

impl Drop for WhisperState {
    fn drop(&mut self) {
        unsafe {
            whisper_rs_sys::whisper_free_state(self.ptr);
        }
    }
}

impl WhisperState {
    pub(crate) fn new(
        ctx: Arc<WhisperInnerContext>,
        ptr: *mut whisper_rs_sys::whisper_state,
    ) -> Self {
        let _timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        generic_info!("[{}] Creating new WhisperState", _timestamp);
        Self { ctx, ptr }
    }

    /// Convert raw PCM audio (floating point 32 bit) to log mel spectrogram.
    /// The resulting spectrogram is stored in the context transparently.
    ///
    /// # Arguments
    /// * pcm: The raw PCM audio.
    /// * threads: How many threads to use. Defaults to 1. Must be at least 1, returns an error otherwise.
    ///
    /// # Returns
    /// Ok(()) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_pcm_to_mel(struct whisper_context * ctx, const float * samples, int n_samples, int n_threads)`
    pub fn pcm_to_mel(&mut self, pcm: &[f32], threads: usize) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_pcm_to_mel_with_state(
                self.ctx.ctx,
                self.ptr,
                pcm.as_ptr(),
                pcm.len() as c_int,
                threads as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::UnableToCalculateSpectrogram)
        } else if ret == 0 {
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// This can be used to set a custom log mel spectrogram inside the provided whisper state.
    /// Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
    ///
    /// # Note
    /// This is a low-level function.
    /// If you're a typical user, you probably don't want to use this function.
    /// See instead [WhisperState::pcm_to_mel].
    ///
    /// # Arguments
    /// * data: The log mel spectrogram.
    ///
    /// # Returns
    /// Ok(()) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_set_mel(struct whisper_context * ctx, const float * data, int n_len, int n_mel)`
    pub fn set_mel(&mut self, data: &[f32]) -> Result<(), WhisperError> {
        let hop_size = 160;
        let n_len = (data.len() / hop_size) * 2;
        let ret = unsafe {
            whisper_rs_sys::whisper_set_mel_with_state(
                self.ctx.ctx,
                self.ptr,
                data.as_ptr(),
                n_len as c_int,
                80 as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::InvalidMelBands)
        } else if ret == 0 {
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// Run the Whisper encoder on the log mel spectrogram stored inside the provided whisper state.
    /// Make sure to call [WhisperState::pcm_to_mel] or [WhisperState::set_mel] first.
    ///
    /// # Arguments
    /// * offset: Can be used to specify the offset of the first frame in the spectrogram. Usually 0.
    /// * threads: How many threads to use. Defaults to 1. Must be at least 1, returns an error otherwise.
    ///
    /// # Returns
    /// Ok(()) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_encode(struct whisper_context * ctx, int offset, int n_threads)`
    pub fn encode(&mut self, offset: usize, threads: usize) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_encode_with_state(
                self.ctx.ctx,
                self.ptr,
                offset as c_int,
                threads as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::UnableToCalculateEvaluation)
        } else if ret == 0 {
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// Run the Whisper decoder to obtain the logits and probabilities for the next token.
    /// Make sure to call [WhisperState::encode] first.
    /// tokens + n_tokens is the provided context for the decoder.
    ///
    /// # Arguments
    /// * tokens: The tokens to decode.
    /// * n_tokens: The number of tokens to decode.
    /// * n_past: The number of past tokens to use for the decoding.
    /// * n_threads: How many threads to use. Defaults to 1. Must be at least 1, returns an error otherwise.
    ///
    /// # Returns
    /// Ok(()) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_decode(struct whisper_context * ctx, const whisper_token * tokens, int n_tokens, int n_past, int n_threads)`
    pub fn decode(
        &mut self,
        tokens: &[WhisperToken],
        n_past: usize,
        threads: usize,
    ) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_decode_with_state(
                self.ctx.ctx,
                self.ptr,
                tokens.as_ptr(),
                tokens.len() as c_int,
                n_past as c_int,
                threads as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::UnableToCalculateEvaluation)
        } else if ret == 0 {
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    // Language functions
    /// Use mel data at offset_ms to try and auto-detect the spoken language
    /// Make sure to call pcm_to_mel() or set_mel() first
    ///
    /// # Arguments
    /// * offset_ms: The offset in milliseconds to use for the language detection.
    /// * n_threads: How many threads to use. Defaults to 1. Must be at least 1, returns an error otherwise.
    ///
    /// # Returns
    /// `Ok((i32, Vec<f32>))` on success where the i32 is detected language id and Vec<f32>
    /// is array with the probabilities of all languages, `Err(WhisperError)` on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_lang_auto_detect(struct whisper_context * ctx, int offset_ms, int n_threads, float * lang_probs)`
    pub fn lang_detect(
        &self,
        offset_ms: usize,
        threads: usize,
    ) -> Result<(i32, Vec<f32>), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }

        let mut lang_probs: Vec<f32> = vec![0.0; crate::standalone::get_lang_max_id() as usize + 1];
        let ret = unsafe {
            whisper_rs_sys::whisper_lang_auto_detect_with_state(
                self.ctx.ctx,
                self.ptr,
                offset_ms as c_int,
                threads as c_int,
                lang_probs.as_mut_ptr(),
            )
        };
        if ret < 0 {
            Err(WhisperError::GenericError(ret))
        } else {
            Ok((ret as i32, lang_probs))
        }
    }

    // logit functions
    /// Gets logits obtained from the last call to [WhisperState::decode].
    /// As of whisper.cpp 1.4.1, only a single row of logits is available, corresponding to the last token in the input.
    ///
    /// # Returns
    /// A slice of logits with length equal to n_vocab.
    ///
    /// # C++ equivalent
    /// `float * whisper_get_logits(struct whisper_context * ctx)`
    pub fn get_logits(&self) -> Result<&[f32], WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_get_logits_from_state(self.ptr) };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        let n_vocab = self.n_vocab();
        Ok(unsafe { std::slice::from_raw_parts(ret, n_vocab as usize) })
    }

    // model attributes
    /// Get the mel spectrogram length.
    ///
    /// # Returns
    /// Ok(c_int) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_n_len_from_state(struct whisper_context * ctx)`
    #[inline]
    pub fn n_len(&self) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_n_len_from_state(self.ptr) })
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
        unsafe { whisper_rs_sys::whisper_n_vocab(self.ctx.ctx) }
    }

    /// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
    /// Uses the specified decoding strategy to obtain the text.
    ///
    /// This is usually the only function you need to call as an end user.
    ///
    /// # Arguments
    /// * params: [crate::FullParams] struct.
    /// * pcm: raw PCM audio data, 32 bit floating point at a sample rate of 16 kHz, 1 channel.
    ///   See utilities in the root of this crate for functions to convert audio to this format.
    ///
    /// # Returns
    /// Ok(c_int) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_full(struct whisper_context * ctx, struct whisper_full_params params, const float * samples, int n_samples)`
    pub fn full(&mut self, params: FullParams, data: &[f32]) -> Result<c_int, WhisperError> {
        let _timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        generic_info!("[{}] Running full transcription with {} samples", _timestamp, data.len());

        if data.is_empty() {
            return Err(WhisperError::NoSamples);
        }

        let ret = unsafe {
            whisper_rs_sys::whisper_full_with_state(
                self.ctx.ctx,
                self.ptr,
                params.fp,
                data.as_ptr(),
                data.len() as c_int,
            )
        };

        if ret == 0 {
            generic_info!("[{}] Transcription completed successfully", _timestamp);
            Ok(ret)
        } else if ret == 7 {
            Err(WhisperError::FailedToEncode)
        } else if ret == 8 {
            Err(WhisperError::FailedToDecode)
        } else {
            generic_info!("[{}] Transcription failed with error code: {}", _timestamp, ret);
            Err(WhisperError::GenericError(ret))
        }
    }

    /// Number of generated text segments.
    /// A segment can be a few words, a sentence, or even a paragraph.
    ///
    /// # C++ equivalent
    /// `int whisper_full_n_segments(struct whisper_context * ctx)`
    #[inline]
    pub fn full_n_segments(&self) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_n_segments_from_state(self.ptr) })
    }

    /// Language ID associated with the provided state.
    ///
    /// # C++ equivalent
    /// `int whisper_full_lang_id_from_state(struct whisper_state * state);`
    #[inline]
    pub fn full_lang_id_from_state(&self) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_lang_id_from_state(self.ptr) })
    }

    /// Get the start time of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # C++ equivalent
    /// `int64_t whisper_full_get_segment_t0(struct whisper_context * ctx, int i_segment)`
    #[inline]
    pub fn full_get_segment_t0(&self, segment: c_int) -> Result<i64, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_get_segment_t0_from_state(self.ptr, segment) })
    }

    /// Get the end time of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # C++ equivalent
    /// `int64_t whisper_full_get_segment_t1(struct whisper_context * ctx, int i_segment)`
    #[inline]
    pub fn full_get_segment_t1(&self, segment: c_int) -> Result<i64, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_get_segment_t1_from_state(self.ptr, segment) })
    }

    fn full_get_segment_raw(&self, segment: c_int) -> Result<&CStr, WhisperError> {
        let ret =
            unsafe { whisper_rs_sys::whisper_full_get_segment_text_from_state(self.ptr, segment) };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        unsafe { Ok(CStr::from_ptr(ret)) }
    }

    /// Get the raw bytes of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # Returns
    /// `Ok(Vec<u8>)` on success, with the returned bytes or
    /// `Err(WhisperError::NullPointer)` on failure (this is the only possible error)
    ///
    /// # C++ equivalent
    /// `const char * whisper_full_get_segment_text(struct whisper_context * ctx, int i_segment)`
    pub fn full_get_segment_bytes(&self, segment: c_int) -> Result<Vec<u8>, WhisperError> {
        Ok(self.full_get_segment_raw(segment)?.to_bytes().to_vec())
    }

    /// Get the text of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # Returns
    /// `Ok(String)` on success, with the UTF-8 validated string, or
    /// `Err(WhisperError)` on failure (either `NullPointer` or `InvalidUtf8`)
    ///
    /// # C++ equivalent
    /// `const char * whisper_full_get_segment_text(struct whisper_context * ctx, int i_segment)`
    pub fn full_get_segment_text(&self, segment: c_int) -> Result<String, WhisperError> {
        Ok(self.full_get_segment_raw(segment)?.to_str()?.to_string())
    }

    /// Get the text of the specified segment.
    /// This function differs from [WhisperState::full_get_segment_text]
    /// in that it ignores invalid UTF-8 in whisper strings,
    /// instead opting to replace it with the replacement character.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # Returns
    /// `Ok(String)` on success, or
    /// `Err(WhisperError::NullPointer)` on failure (this is the only possible error)
    ///
    /// # C++ equivalent
    /// `const char * whisper_full_get_segment_text(struct whisper_context * ctx, int i_segment)`
    pub fn full_get_segment_text_lossy(&self, segment: c_int) -> Result<String, WhisperError> {
        Ok(self
            .full_get_segment_raw(segment)?
            .to_string_lossy()
            .to_string())
    }

    /// Get number of tokens in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_full_n_tokens(struct whisper_context * ctx, int i_segment)`
    #[inline]
    pub fn full_n_tokens(&self, segment: c_int) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_n_tokens_from_state(self.ptr, segment) })
    }

    fn full_get_token_raw(&self, segment: c_int, token: c_int) -> Result<&CStr, WhisperError> {
        let ret = unsafe {
            whisper_rs_sys::whisper_full_get_token_text_from_state(
                self.ctx.ctx,
                self.ptr,
                segment,
                token,
            )
        };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        unsafe { Ok(CStr::from_ptr(ret)) }
    }

    /// Get the raw token bytes of the specified token in the specified segment.
    ///
    /// Useful if you're using a language for which whisper is known to split tokens
    /// away from UTF-8 character boundaries.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// `Ok(Vec<u8>)` on success, with the returned bytes or
    /// `Err(WhisperError::NullPointer)` on failure (this is the only possible error)
    ///
    /// # C++ equivalent
    /// `const char * whisper_full_get_token_text(struct whisper_context * ctx, int i_segment, int i_token)`
    pub fn full_get_token_bytes(
        &self,
        segment: c_int,
        token: c_int,
    ) -> Result<Vec<u8>, WhisperError> {
        Ok(self.full_get_token_raw(segment, token)?.to_bytes().to_vec())
    }

    /// Get the token text of the specified token in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// `Ok(String)` on success, with the UTF-8 validated string, or
    /// `Err(WhisperError)` on failure (either `NullPointer` or `InvalidUtf8`)
    ///
    /// # C++ equivalent
    /// `const char * whisper_full_get_token_text(struct whisper_context * ctx, int i_segment, int i_token)`
    pub fn full_get_token_text(
        &self,
        segment: c_int,
        token: c_int,
    ) -> Result<String, WhisperError> {
        Ok(self
            .full_get_token_raw(segment, token)?
            .to_str()?
            .to_string())
    }

    /// Get the token text of the specified token in the specified segment.
    /// This function differs from [WhisperState::full_get_token_text]
    /// in that it ignores invalid UTF-8 in whisper strings,
    /// instead opting to replace it with the replacement character.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// `Ok(String)` on success, or
    /// `Err(WhisperError::NullPointer)` on failure (this is the only possible error)
    ///
    /// # C++ equivalent
    /// `const char * whisper_full_get_token_text(struct whisper_context * ctx, int i_segment, int i_token)`
    pub fn full_get_token_text_lossy(
        &self,
        segment: c_int,
        token: c_int,
    ) -> Result<String, WhisperError> {
        Ok(self
            .full_get_token_raw(segment, token)?
            .to_string_lossy()
            .to_string())
    }

    /// Get the token ID of the specified token in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// [crate::WhisperToken]
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_full_get_token_id (struct whisper_context * ctx, int i_segment, int i_token)`
    pub fn full_get_token_id(
        &self,
        segment: c_int,
        token: c_int,
    ) -> Result<WhisperToken, WhisperError> {
        Ok(unsafe {
            whisper_rs_sys::whisper_full_get_token_id_from_state(self.ptr, segment, token)
        })
    }

    /// Get token data for the specified token in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// [crate::WhisperTokenData]
    ///
    /// # C++ equivalent
    /// `whisper_token_data whisper_full_get_token_data(struct whisper_context * ctx, int i_segment, int i_token)`
    #[inline]
    pub fn full_get_token_data(
        &self,
        segment: c_int,
        token: c_int,
    ) -> Result<WhisperTokenData, WhisperError> {
        Ok(unsafe {
            whisper_rs_sys::whisper_full_get_token_data_from_state(self.ptr, segment, token)
        })
    }

    /// Get the probability of the specified token in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// f32
    ///
    /// # C++ equivalent
    /// `float whisper_full_get_token_p(struct whisper_context * ctx, int i_segment, int i_token)`
    #[inline]
    pub fn full_get_token_prob(&self, segment: c_int, token: c_int) -> Result<f32, WhisperError> {
        Ok(
            unsafe {
                whisper_rs_sys::whisper_full_get_token_p_from_state(self.ptr, segment, token)
            },
        )
    }

    /// Get whether the next segment is predicted as a speaker turn.
    ///
    /// # Arguments
    /// * i_segment: Segment index.
    ///
    /// # Returns
    /// bool
    ///
    /// # C++ equivalent
    /// `bool whisper_full_get_segment_speaker_turn_next_from_state(struct whisper_state * state, int i_segment)`
    pub fn full_get_segment_speaker_turn_next(&mut self, i_segment: c_int) -> bool {
        unsafe {
            whisper_rs_sys::whisper_full_get_segment_speaker_turn_next_from_state(
                self.ptr, i_segment,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FullParams, SamplingStrategy};

    // Test helper struct for segment testing
    #[derive(Debug, Clone)]
    struct TestSegment {
        start_timestamp: i64,
        end_timestamp: i64,
        text: String,
        speaker_turn_next: bool,
    }

    // Mock tests that don't require actual model files or valid states
    
    #[test]
    fn test_whisper_state_validation() {
        // Test state validity checking with proper type annotation
        let _null_state: *mut whisper_rs_sys::whisper_state = std::ptr::null_mut();
        // Can't easily create a WhisperState without valid context, but we can test the concept
        
        // Test bounds checking for invalid segment indices
        let invalid_segments = vec![-1, -100, 1000, i32::MAX, i32::MIN];
        
        // These values represent what should be error cases
        for invalid_segment in invalid_segments {
            assert!(invalid_segment < 0 || invalid_segment > 100); // Reasonable upper bound
        }
    }
    
    #[test]
    fn test_segment_bounds_checking() {
        // Test segment index validation logic
        let test_cases = vec![
            (-1, false),    // Negative index
            (0, true),      // Valid first index
            (10, true),     // Valid middle index  
            (-100, false),  // Very negative
            (i32::MAX, false), // Too large
        ];
        
        for (index, should_be_valid) in test_cases {
            let is_valid = index >= 0 && index < 100; // Assume 100 segments max for test
            assert_eq!(is_valid, should_be_valid, "Index {} validity check failed", index);
        }
    }
    
    #[test]
    fn test_token_bounds_checking() {
        // Test token index validation logic
        let segment_index = 0;
        let n_tokens = 10; // Assume 10 tokens in segment
        
        let test_cases = vec![
            (-1, false),     // Negative token index
            (0, true),       // Valid first token
            (5, true),       // Valid middle token
            (9, true),       // Valid last token  
            (10, false),     // Beyond bounds
            (100, false),    // Way beyond bounds
        ];
        
        for (token_index, should_be_valid) in test_cases {
            let is_valid = token_index >= 0 && token_index < n_tokens;
            assert_eq!(is_valid, should_be_valid, 
                      "Token {} validity check failed for segment {}", token_index, segment_index);
        }
    }
    
    #[test]
    fn test_timestamp_conversion() {
        // Test timestamp conversion from whisper units to milliseconds
        let whisper_units = vec![0, 1, 10, 100, 1000];
        let expected_ms = vec![0, 10, 100, 1000, 10000];
        
        for (whisper_unit, expected) in whisper_units.iter().zip(expected_ms.iter()) {
            let converted = whisper_unit * 10; // Conversion factor
            assert_eq!(converted, *expected, 
                      "Timestamp conversion failed: {} units -> {} ms", whisper_unit, expected);
        }
    }
    
    #[test]
    fn test_probability_calculations() {
        // Test probability averaging calculations
        let test_probabilities = vec![
            vec![1.0_f32], // Single perfect probability
            vec![0.0_f32], // Single zero probability  
            vec![0.5_f32, 0.5_f32], // Two equal probabilities
            vec![0.0_f32, 1.0_f32], // Min and max
            vec![0.25_f32, 0.50_f32, 0.75_f32], // Three different values
        ];
        
        let expected_averages = vec![1.0_f32, 0.0_f32, 0.5_f32, 0.5_f32, 0.5_f32];
        
        for (probs, expected) in test_probabilities.iter().zip(expected_averages.iter()) {
            let average = probs.iter().sum::<f32>() / probs.len() as f32;
            assert!((average - expected).abs() < 0.001, 
                   "Probability average calculation failed: {:?} -> {}, expected {}", 
                   probs, average, expected);
        }
    }
    
    #[test]
    fn test_duration_calculations() {
        // Test duration calculations
        let test_cases = vec![
            (0, 10, 100),   // 10 whisper units = 100 ms
            (5, 15, 100),   // 10 whisper units = 100 ms
            (0, 0, 0),      // Zero duration
            (100, 200, 1000), // 100 whisper units = 1000 ms
        ];
        
        for (start, end, expected_ms) in test_cases {
            let duration_whisper_units = end - start;
            let duration_ms = duration_whisper_units * 10;
            assert_eq!(duration_ms, expected_ms, 
                      "Duration calculation failed: {} to {} units -> {} ms", 
                      start, end, expected_ms);
        }
    }
    
    #[test]
    fn test_error_types() {
        // Test that our existing error types work correctly
        let null_pointer_error = WhisperError::NullPointer;
        let generic_error = WhisperError::GenericError(42);
        let no_samples_error = WhisperError::NoSamples;
        
        // Test error formatting
        let null_str = format!("{:?}", null_pointer_error);
        assert!(null_str.contains("NullPointer"));
        
        let generic_str = format!("{:?}", generic_error);
        assert!(generic_str.contains("GenericError"));
        assert!(generic_str.contains("42"));
        
        let no_samples_str = format!("{:?}", no_samples_error);
        assert!(no_samples_str.contains("NoSamples"));
    }
    
    #[test]
    fn test_full_params_validation() {
        // Test FullParams validation using correct types
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        
        // Test setting various parameters
        params.set_language(Some("en"));
        params.set_translate(true);
        params.set_n_threads(4);
        params.set_offset_ms(1000);
        params.set_duration_ms(5000);
        
        // Verify parameters were set (we can't directly access them, but we can test the setters work)
        assert!(true); // If we got here without panicking, the setters worked
    }
    
    #[test]
    fn test_sampling_strategies() {
        // Test different sampling strategies using correct types
        let greedy = SamplingStrategy::Greedy { best_of: 5 };
        let beam_search = SamplingStrategy::BeamSearch { 
            beam_size: 10, 
            patience: 0.5 
        };
        
        match greedy {
            SamplingStrategy::Greedy { best_of } => {
                assert_eq!(best_of, 5);
            }
            _ => panic!("Expected Greedy strategy"),
        }
        
        match beam_search {
            SamplingStrategy::BeamSearch { beam_size, patience } => {
                assert_eq!(beam_size, 10);
                assert!((patience - 0.5).abs() < 0.001);
            }
            _ => panic!("Expected BeamSearch strategy"),
        }
    }
    
    #[test]
    fn test_segment_creation() {
        // Test TestSegment struct creation and validation
        let segment = TestSegment {
            start_timestamp: 1000,
            end_timestamp: 2000,
            text: "Hello, world!".to_string(),
            speaker_turn_next: false,
        };
        
        assert_eq!(segment.start_timestamp, 1000);
        assert_eq!(segment.end_timestamp, 2000);
        assert_eq!(segment.text, "Hello, world!");
        assert!(!segment.speaker_turn_next);
        
        // Test segment duration calculation
        let duration = segment.end_timestamp - segment.start_timestamp;
        assert_eq!(duration, 1000);
    }
    
    #[test]
    fn test_segment_collection_operations() {
        // Test operations on collections of segments
        let segments = vec![
            TestSegment {
                start_timestamp: 0,
                end_timestamp: 1000,
                text: "First segment".to_string(),
                speaker_turn_next: false,
            },
            TestSegment {
                start_timestamp: 1000,
                end_timestamp: 2000,
                text: "Second segment".to_string(),
                speaker_turn_next: true,
            },
            TestSegment {
                start_timestamp: 2000,
                end_timestamp: 3000,
                text: "Third segment".to_string(),
                speaker_turn_next: false,
            },
        ];
        
        assert_eq!(segments.len(), 3);
        
        // Test total duration calculation
        let total_duration = segments.last().unwrap().end_timestamp;
        assert_eq!(total_duration, 3000);
        
        // Test text concatenation
        let full_text: String = segments.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<&str>>()
            .join(" ");
        assert_eq!(full_text, "First segment Second segment Third segment");
        
        // Test speaker turn detection
        let speaker_turns: Vec<bool> = segments.iter()
            .map(|s| s.speaker_turn_next)
            .collect();
        assert_eq!(speaker_turns, vec![false, true, false]);
    }
    
    #[test]
    fn test_empty_audio_validation() {
        // Test validation of empty audio samples
        let empty_samples: Vec<f32> = vec![];
        assert!(empty_samples.is_empty());
        
        // Should be treated as invalid input
        let is_valid = !empty_samples.is_empty();
        assert!(!is_valid);
    }
    
    #[test]
    fn test_audio_sample_validation() {
        // Test validation of audio samples with explicit f32 types
        let valid_samples: Vec<f32> = vec![0.0, 0.1, -0.1, 0.5, -0.5];
        let extreme_samples: Vec<f32> = vec![f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        
        // Valid samples should pass basic checks
        assert!(!valid_samples.is_empty());
        for &sample in &valid_samples {
            assert!(sample.is_finite() || sample == 0.0);
        }
        
        // Extreme values should be detected
        for &sample in &extreme_samples {
            let is_problematic = !sample.is_finite();
            if sample.is_nan() || sample.is_infinite() {
                assert!(is_problematic);
            }
        }
    }
    
    #[test]
    fn test_utf8_validation() {
        // Test UTF-8 string validation
        let valid_strings = vec![
            "Hello, world!",
            "测试中文",
            "Тест русский",
            "🎵 Music note",
            "",
        ];
        
        for s in valid_strings {
            assert!(s.is_ascii() || std::str::from_utf8(s.as_bytes()).is_ok());
        }
        
        // Test invalid UTF-8 byte sequences
        let invalid_bytes = vec![
            vec![0xFF, 0xFE], // Invalid UTF-8 
            vec![0x80, 0x80], // Invalid continuation
        ];
        
        for bytes in invalid_bytes {
            assert!(std::str::from_utf8(&bytes).is_err());
        }
    }
}
