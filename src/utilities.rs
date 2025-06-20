use crate::WhisperError;
use crate::common_logging::generic_info;
use std::time::SystemTime;

/// Convert an array of 16 bit mono audio samples to a vector of 32 bit floats.
///
/// # Arguments
/// * `samples` - The array of 16 bit mono audio samples.
/// * `output` - The vector of 32 bit floats to write the converted samples to.
///
/// # Panics
/// * if `samples.len != output.len()`
///
/// # Examples
/// ```
/// # use whisper_rs::convert_integer_to_float_audio;
/// let samples = [0i16; 1024];
/// let mut output = vec![0.0f32; samples.len()];
/// convert_integer_to_float_audio(&samples, &mut output).expect("input and output lengths should be equal");
/// ```
pub fn convert_integer_to_float_audio(
    samples: &[i16],
    output: &mut [f32],
) -> Result<(), WhisperError> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    if samples.len() != output.len() {
        generic_info!("[{}] Length mismatch: input={}, output={}", timestamp, samples.len(), output.len());
        return Err(WhisperError::InputOutputLengthMismatch {
            input_len: samples.len(),
            output_len: output.len(),
        });
    }

    generic_info!("[{}] Converting {} i16 samples to f32", timestamp, samples.len());

    for (input, output) in samples.iter().zip(output.iter_mut()) {
        *output = *input as f32 / 32768.0;
    }

    Ok(())
}

/// Convert 32-bit floating point stereo PCM audio to 32-bit floating point mono PCM audio.
///
/// # Arguments
/// * `samples` - The array of 32-bit floating point stereo PCM audio samples.
///
/// # Errors
/// * if `samples.len()` is odd
///
/// # Returns
/// A vector of 32-bit floating point mono PCM audio samples.
///
/// # Examples
/// ```
/// # use whisper_rs::convert_stereo_to_mono_audio;
/// let samples = [0.0f32; 1024];
/// let mono = convert_stereo_to_mono_audio(&samples).expect("should be no half samples missing");
/// ```
pub fn convert_stereo_to_mono_audio(samples: &[f32]) -> Result<Vec<f32>, WhisperError> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    if samples.len() & 1 != 0 {
        generic_info!("[{}] Odd number of samples: {}", timestamp, samples.len());
        return Err(WhisperError::HalfSampleMissing(samples.len()));
    }

    generic_info!("[{}] Converting {} stereo samples to {} mono samples", 
                 timestamp, samples.len(), samples.len() / 2);

    Ok(samples
        .chunks_exact(2)
        .map(|x| (x[0] + x[1]) / 2.0)
        .collect())
}

/// Convert 16-bit stereo audio directly to mono f32 audio
///
/// # Arguments
/// * `samples` - The array of 16-bit stereo PCM audio samples
///
/// # Errors
/// * if `samples.len()` is odd
///
/// # Returns
/// A vector of 32-bit floating point mono PCM audio samples
pub fn convert_stereo_i16_to_mono_f32(samples: &[i16]) -> Result<Vec<f32>, WhisperError> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    if samples.len() & 1 != 0 {
        generic_info!("[{}] Odd number of i16 samples: {}", timestamp, samples.len());
        return Err(WhisperError::HalfSampleMissing(samples.len()));
    }

    generic_info!("[{}] Converting {} stereo i16 samples to {} mono f32 samples", 
                 timestamp, samples.len(), samples.len() / 2);

    Ok(samples
        .chunks_exact(2)
        .map(|x| ((x[0] as f32 + x[1] as f32) / 2.0) / 32768.0)
        .collect())
}

/// Normalize audio samples to a specific peak level
///
/// # Arguments
/// * `samples` - The audio samples to normalize
/// * `peak_level` - The target peak level (0.0 to 1.0)
///
/// # Returns
/// Normalized audio samples
pub fn normalize_audio(samples: &mut [f32], peak_level: f32) -> Result<(), WhisperError> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if !(0.0..=1.0).contains(&peak_level) {
        return Err(WhisperError::GenericError(-1));
    }

    if samples.is_empty() {
        return Ok(());
    }

    let max_sample = samples.iter()
        .map(|&x| x.abs())
        .fold(0.0f32, f32::max);

    if max_sample > 0.0 {
        let scale = peak_level / max_sample;
        generic_info!("[{}] Normalizing {} samples with scale factor {}", 
                     timestamp, samples.len(), scale);
        
        for sample in samples.iter_mut() {
            *sample *= scale;
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_integer_to_float_conversion() {
        let samples = [32767i16, -32768i16, 0i16];
        let mut output = vec![0.0f32; samples.len()];
        
        convert_integer_to_float_audio(&samples, &mut output).unwrap();
        
        assert!((output[0] - 0.999969482).abs() < 0.0001); // 32767 / 32768
        assert_eq!(output[1], -1.0); // -32768 / 32768
        assert_eq!(output[2], 0.0); // 0 / 32768
    }

    #[test]
    fn test_integer_to_float_length_mismatch() {
        let samples = [0i16; 5];
        let mut output = vec![0.0f32; 3]; // Different length
        
        let result = convert_integer_to_float_audio(&samples, &mut output);
        assert!(result.is_err());
        
        if let Err(WhisperError::InputOutputLengthMismatch { input_len, output_len }) = result {
            assert_eq!(input_len, 5);
            assert_eq!(output_len, 3);
        } else {
            panic!("Expected InputOutputLengthMismatch error");
        }
    }

    #[test]
    fn test_integer_to_float_empty() {
        let samples: [i16; 0] = [];
        let mut output = vec![];
        
        let result = convert_integer_to_float_audio(&samples, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_integer_to_float_extreme_values() {
        let samples = [i16::MAX, i16::MIN, 0, 1, -1];
        let mut output = vec![0.0f32; samples.len()];
        
        convert_integer_to_float_audio(&samples, &mut output).unwrap();
        
        assert!((output[0] - (i16::MAX as f32 / 32768.0)).abs() < 0.0001);
        assert_eq!(output[1], -1.0); // i16::MIN / 32768
        assert_eq!(output[2], 0.0);
        assert!((output[3] - (1.0 / 32768.0)).abs() < 0.0001);
        assert!((output[4] - (-1.0 / 32768.0)).abs() < 0.0001);
    }

    #[test]
    fn test_stereo_to_mono_conversion() {
        let samples = [0.5f32, 0.3f32, -0.2f32, 0.8f32];
        let mono = convert_stereo_to_mono_audio(&samples).unwrap();
        
        assert_eq!(mono.len(), 2);
        assert_eq!(mono[0], 0.4); // (0.5 + 0.3) / 2
        assert_eq!(mono[1], 0.3); // (-0.2 + 0.8) / 2
    }

    #[test]
    pub fn assert_stereo_to_mono_err() {
        // Create an odd number of samples to trigger an error
        let samples = vec![0.0f32; 1023]; // Odd number
        let mono = convert_stereo_to_mono_audio(&samples);
        assert!(mono.is_err());
        
        if let Err(WhisperError::HalfSampleMissing(len)) = mono {
            assert_eq!(len, 1023);
        } else {
            panic!("Expected HalfSampleMissing error");
        }
    }

    #[test]
    fn test_stereo_to_mono_empty() {
        let samples: [f32; 0] = [];
        let mono = convert_stereo_to_mono_audio(&samples).unwrap();
        assert_eq!(mono.len(), 0);
    }

    #[test]
    fn test_stereo_to_mono_single_pair() {
        let samples = [1.0f32, -1.0f32];
        let mono = convert_stereo_to_mono_audio(&samples).unwrap();
        assert_eq!(mono.len(), 1);
        assert_eq!(mono[0], 0.0); // (1.0 + (-1.0)) / 2
    }

    #[test]
    fn test_stereo_to_mono_extreme_values() {
        let samples = [f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY];
        let mono = convert_stereo_to_mono_audio(&samples).unwrap();
        assert_eq!(mono.len(), 2);
        // (f32::MAX + f32::MIN) / 2 should be approximately 0.0
        assert!(mono[0].is_finite() && mono[0].abs() < 1e30);
        // (f32::INFINITY + f32::NEG_INFINITY) / 2 should be NaN (undefined operation)
        assert!(mono[1].is_nan());
    }

    #[test]
    fn test_stereo_i16_to_mono_f32() {
        let samples = [1000i16, 2000i16, -1000i16, -2000i16];
        let mono = convert_stereo_i16_to_mono_f32(&samples).unwrap();
        
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - ((1000.0 + 2000.0) / 2.0 / 32768.0)).abs() < 0.0001);
        assert!((mono[1] - ((-1000.0 + -2000.0) / 2.0 / 32768.0)).abs() < 0.0001);
    }

    #[test]
    fn test_stereo_i16_to_mono_f32_odd_length() {
        let samples = [1000i16, 2000i16, 3000i16]; // Odd length
        let result = convert_stereo_i16_to_mono_f32(&samples);
        assert!(result.is_err());
        
        if let Err(WhisperError::HalfSampleMissing(len)) = result {
            assert_eq!(len, 3);
        } else {
            panic!("Expected HalfSampleMissing error");
        }
    }

    #[test]
    fn test_stereo_i16_to_mono_f32_empty() {
        let samples: [i16; 0] = [];
        let mono = convert_stereo_i16_to_mono_f32(&samples).unwrap();
        assert_eq!(mono.len(), 0);
    }

    #[test]
    fn test_normalize_audio_basic() {
        let mut samples = [0.5f32, -0.8f32, 0.2f32, -0.3f32];
        normalize_audio(&mut samples, 1.0).unwrap();
        
        // The maximum absolute value was 0.8, so scale factor should be 1.0/0.8 = 1.25
        let scale = 1.0 / 0.8;
        assert!((samples[0] - 0.5 * scale).abs() < 0.0001);
        assert!((samples[1] - (-0.8 * scale)).abs() < 0.0001);
        assert!((samples[2] - 0.2 * scale).abs() < 0.0001);
        assert!((samples[3] - (-0.3 * scale)).abs() < 0.0001);
        
        // Check that max absolute value is now 1.0
        let max_abs = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((max_abs - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_normalize_audio_half_level() {
        let mut samples = [1.0f32, -1.0f32, 0.5f32];
        normalize_audio(&mut samples, 0.5).unwrap();
        
        // Max was 1.0, scaling to 0.5 means scale factor is 0.5
        assert!((samples[0] - 0.5).abs() < 0.0001);
        assert!((samples[1] - (-0.5)).abs() < 0.0001);
        assert!((samples[2] - 0.25).abs() < 0.0001);
    }

    #[test]
    fn test_normalize_audio_invalid_peak_level() {
        let mut samples = [0.5f32, -0.5f32];
        
        // Test peak level > 1.0
        let result = normalize_audio(&mut samples, 1.5);
        assert!(result.is_err());
        
        // Test negative peak level
        let result = normalize_audio(&mut samples, -0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_audio_empty() {
        let mut samples: [f32; 0] = [];
        let result = normalize_audio(&mut samples, 1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_normalize_audio_zero_samples() {
        let mut samples = [0.0f32; 5];
        let result = normalize_audio(&mut samples, 1.0);
        assert!(result.is_ok());
        
        // All samples should remain zero
        for &sample in &samples {
            assert_eq!(sample, 0.0);
        }
    }

    #[test]
    fn test_normalize_audio_very_small_values() {
        let mut samples = [0.0001f32, -0.0002f32, 0.00015f32];
        normalize_audio(&mut samples, 1.0).unwrap();
        
        // Should scale up properly
        let max_abs = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((max_abs - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_large_audio_conversion() {
        // Test with larger arrays to ensure performance
        let size = 44100; // 1 second at 44.1kHz
        let samples: Vec<i16> = (0..size).map(|i| ((i as f32 * 0.1).sin() * 16384.0) as i16).collect();
        let mut output = vec![0.0f32; size];
        
        let result = convert_integer_to_float_audio(&samples, &mut output);
        assert!(result.is_ok());
        
        // Check that conversion maintains signal characteristics
        for (i, (&input, &output_val)) in samples.iter().zip(output.iter()).enumerate() {
            let expected = input as f32 / 32768.0;
            assert!((output_val - expected).abs() < 0.0001, 
                   "Mismatch at index {}: expected {}, got {}", i, expected, output_val);
        }
    }

    #[test]
    fn test_large_stereo_conversion() {
        // Test stereo conversion with larger arrays
        let size = 44100 * 2; // 1 second stereo at 44.1kHz
        let samples: Vec<f32> = (0..size).map(|i| {
            if i % 2 == 0 { 0.5 } else { -0.5 } // Alternating left/right channels
        }).collect();
        
        let mono = convert_stereo_to_mono_audio(&samples).unwrap();
        assert_eq!(mono.len(), size / 2);
        
        // All mono samples should be 0.0 (average of 0.5 and -0.5)
        for &sample in &mono {
            assert!((sample - 0.0).abs() < 0.0001);
        }
    }

    #[test]
    fn test_conversion_round_trip() {
        // Test converting from i16 to f32 and ensuring we can detect the original values
        let original = [1000i16, -2000i16, 32767i16, -32768i16, 0i16];
        let mut float_version = vec![0.0f32; original.len()];
        
        convert_integer_to_float_audio(&original, &mut float_version).unwrap();
        
        // Convert back to i16 (lossy due to floating point precision)
        let recovered: Vec<i16> = float_version.iter()
            .map(|&f| (f * 32768.0) as i16)
            .collect();
        
        // Should be very close to original (within 1 due to rounding)
        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            let diff = (orig - rec).abs();
            assert!(diff <= 1, "Index {}: original {}, recovered {}, diff {}", i, orig, rec, diff);
        }
    }

    #[test]
    fn test_audio_processing_chain() {
        // Test a complete processing chain: stereo i16 -> mono f32 -> normalized
        let stereo_i16 = [8000i16, 12000i16, -4000i16, -6000i16, 16000i16, 24000i16];
        
        // Convert to mono f32
        let mut mono_f32 = convert_stereo_i16_to_mono_f32(&stereo_i16).unwrap();
        assert_eq!(mono_f32.len(), 3);
        
        // Normalize
        normalize_audio(&mut mono_f32, 0.8).unwrap();
        
        // Check that max absolute value is approximately 0.8
        let max_abs = mono_f32.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((max_abs - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_edge_case_values() {
        // Test with special floating point values
        let mut samples = [f32::MIN_POSITIVE, f32::EPSILON, 1.0 - f32::EPSILON];
        let original = samples.clone();
        
        normalize_audio(&mut samples, 1.0).unwrap();
        
        // The largest value should become 1.0
        let max_abs = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((max_abs - 1.0).abs() < 0.0001);
        
        // Relative ratios should be preserved
        let scale = 1.0 / original[2]; // Scale factor
        for (i, (&orig, &scaled)) in original.iter().zip(samples.iter()).enumerate() {
            assert!((scaled - orig * scale).abs() < 0.0001, 
                   "Index {}: expected {}, got {}", i, orig * scale, scaled);
        }
    }
}
