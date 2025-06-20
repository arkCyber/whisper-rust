/**
 * Enhanced Basic Whisper-RS Usage Example
 * 
 * This example demonstrates the core functionality of whisper-rs for speech-to-text transcription.
 * It loads a Whisper model and processes a WAV audio file to generate timestamped transcriptions.
 * 
 * Features:
 * - Model loading with proper error handling
 * - Audio format conversion (stereo to mono, integer to float)
 * - Transcription with segment timestamps
 * - Comprehensive logging with timestamps
 * - Enhanced error handling and validation
 * 
 * Usage:
 *   wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
 *   wget https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav
 *   cargo run --example basic_use ggml-tiny.bin jfk.wav
 * 
 * Dependencies:
 *   - whisper-rs: Main library for Whisper functionality
 *   - hound: WAV file reading and processing
 * 
 * Author: Enhanced for whisper-rs project
 * Date: 2024
 * License: Same as whisper-rs project
 */

use std::time::SystemTime;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/**
 * Print a timestamped log message to stdout
 * 
 * @param level - Log level (INFO, WARN, ERROR, etc.)
 * @param message - The message to log
 */
fn log_with_timestamp(level: &str, message: &str) {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    println!("[{}] {}: {}", timestamp, level, message);
}

/**
 * Validate command line arguments
 * 
 * @return Result containing (model_path, wav_path) or error message
 */
fn validate_arguments() -> Result<(String, String), String> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", "Validating command line arguments");
    
    let model_path = std::env::args()
        .nth(1)
        .ok_or_else(|| {
            log_with_timestamp("ERROR", "Missing model path argument");
            "Please specify path to model as first argument".to_string()
        })?;
    
    let wav_path = std::env::args()
        .nth(2)
        .ok_or_else(|| {
            log_with_timestamp("ERROR", "Missing WAV file path argument");
            "Please specify path to wav file as second argument".to_string()
        })?;
    
    // Validate file existence
    if !std::path::Path::new(&model_path).exists() {
        let error_msg = format!("Model file does not exist: {}", model_path);
        log_with_timestamp("ERROR", &error_msg);
        return Err(error_msg);
    }
    
    if !std::path::Path::new(&wav_path).exists() {
        let error_msg = format!("WAV file does not exist: {}", wav_path);
        log_with_timestamp("ERROR", &error_msg);
        return Err(error_msg);
    }
    
    log_with_timestamp("INFO", &format!("Model path: {}", model_path));
    log_with_timestamp("INFO", &format!("WAV path: {}", wav_path));
    
    Ok((model_path, wav_path))
}

/**
 * Load and validate WAV audio file
 * 
 * @param wav_path - Path to the WAV file
 * @return Result containing audio samples or error message
 */
fn load_audio_samples(wav_path: &str) -> Result<Vec<i16>, String> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", &format!("Loading audio file: {}", wav_path));
    
    let wav_reader = hound::WavReader::open(wav_path)
        .map_err(|e| {
            let error_msg = format!("Failed to open WAV file: {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    let spec = wav_reader.spec();
    log_with_timestamp("INFO", &format!("Audio specs - Sample rate: {}Hz, Channels: {}, Bits per sample: {}", 
                                       spec.sample_rate, spec.channels, spec.bits_per_sample));
    
    let samples: Result<Vec<i16>, _> = wav_reader
        .into_samples::<i16>()
        .collect();
    
    let samples = samples.map_err(|e| {
        let error_msg = format!("Failed to read audio samples: {}", e);
        log_with_timestamp("ERROR", &error_msg);
        error_msg
    })?;
    
    log_with_timestamp("INFO", &format!("Successfully loaded {} audio samples", samples.len()));
    
    if samples.is_empty() {
        let error_msg = "Audio file contains no samples";
        log_with_timestamp("ERROR", error_msg);
        return Err(error_msg.to_string());
    }
    
    Ok(samples)
}

/**
 * Initialize Whisper context and state
 * 
 * @param model_path - Path to the Whisper model file
 * @return Result containing (context, state) or error message
 */
fn initialize_whisper_context(model_path: &str) -> Result<(WhisperContext, whisper_rs::WhisperState), String> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", &format!("Loading Whisper model: {}", model_path));
    
    let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .map_err(|e| {
            let error_msg = format!("Failed to load model: {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    log_with_timestamp("INFO", "Whisper model loaded successfully");
    
    let state = ctx.create_state().map_err(|e| {
        let error_msg = format!("Failed to create Whisper state: {}", e);
        log_with_timestamp("ERROR", &error_msg);
        error_msg
    })?;
    
    log_with_timestamp("INFO", "Whisper state created successfully");
    
    Ok((ctx, state))
}

/**
 * Configure Whisper parameters for transcription
 * 
 * @param language - Target language for transcription
 * @return Configured FullParams for Whisper
 */
fn configure_whisper_params(language: &str) -> FullParams {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", &format!("Configuring Whisper parameters for language: {}", language));
    
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    
    // Set the language for transcription
    params.set_language(Some(language));
    
    // Disable console output to keep our logs clean
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    
    log_with_timestamp("INFO", "Whisper parameters configured successfully");
    
    params
}

/**
 * Process audio samples for Whisper model compatibility
 * 
 * @param samples - Raw integer audio samples
 * @return Result containing processed f32 mono samples or error message
 */
fn process_audio_samples(samples: &[i16]) -> Result<Vec<f32>, String> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", "Converting audio samples for Whisper compatibility");
    
    // Convert integer samples to float
    let mut inter_samples = vec![0.0f32; samples.len()];
    
    whisper_rs::convert_integer_to_float_audio(samples, &mut inter_samples)
        .map_err(|e| {
            let error_msg = format!("Failed to convert integer to float audio: {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    log_with_timestamp("INFO", "Integer to float conversion completed");
    
    // Convert stereo to mono if needed
    let mono_samples = whisper_rs::convert_stereo_to_mono_audio(&inter_samples)
        .map_err(|e| {
            let error_msg = format!("Failed to convert stereo to mono audio: {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    log_with_timestamp("INFO", &format!("Audio processing completed. Final sample count: {}", mono_samples.len()));
    
    Ok(mono_samples)
}

/**
 * Run transcription and extract results
 * 
 * @param state - Whisper state for processing
 * @param params - Configured Whisper parameters
 * @param samples - Processed audio samples
 * @return Result indicating success or error message
 */
fn run_transcription(
    state: &mut whisper_rs::WhisperState,
    params: FullParams,
    samples: &[f32]
) -> Result<(), String> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", "Starting transcription process");
    
    // Run the Whisper model
    state.full(params, samples).map_err(|e| {
        let error_msg = format!("Failed to run Whisper model: {}", e);
        log_with_timestamp("ERROR", &error_msg);
        error_msg
    })?;
    
    log_with_timestamp("INFO", "Transcription completed successfully");
    
    // Extract and display results
    let num_segments = state.full_n_segments().map_err(|e| {
        let error_msg = format!("Failed to get number of segments: {}", e);
        log_with_timestamp("ERROR", &error_msg);
        error_msg
    })?;
    
    log_with_timestamp("INFO", &format!("Found {} transcription segments", num_segments));
    
    if num_segments == 0 {
        log_with_timestamp("WARN", "No transcription segments found");
        return Ok(());
    }
    
    println!("\n=== TRANSCRIPTION RESULTS ===");
    
    for i in 0..num_segments {
        let segment = state.full_get_segment_text(i).map_err(|e| {
            let error_msg = format!("Failed to get segment text for segment {}: {}", i, e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
        
        let start_timestamp = state.full_get_segment_t0(i).map_err(|e| {
            let error_msg = format!("Failed to get start timestamp for segment {}: {}", i, e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
        
        let end_timestamp = state.full_get_segment_t1(i).map_err(|e| {
            let error_msg = format!("Failed to get end timestamp for segment {}: {}", i, e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
        
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }
    
    println!("=== END TRANSCRIPTION RESULTS ===\n");
    
    Ok(())
}

/**
 * Main function - Enhanced Whisper transcription example
 * 
 * Processes command line arguments, loads audio and model files,
 * performs transcription with comprehensive error handling and logging.
 */
fn main() {
    let start_time = SystemTime::now();
    log_with_timestamp("INFO", "Starting enhanced Whisper transcription example");
    
    // Process execution with comprehensive error handling
    let result = (|| -> Result<(), String> {
        // Validate command line arguments
        let (model_path, wav_path) = validate_arguments()?;
        
        // Load audio samples
        let samples = load_audio_samples(&wav_path)?;
        
        // Initialize Whisper context and state
        let (ctx, mut state) = initialize_whisper_context(&model_path)?;
        
        // Configure transcription parameters
        let language = "en"; // Default to English
        let params = configure_whisper_params(language);
        
        // Process audio samples for Whisper compatibility
        let processed_samples = process_audio_samples(&samples)?;
        
        // Run transcription and display results
        run_transcription(&mut state, params, &processed_samples)?;
        
        Ok(())
    })();
    
    // Handle final result and cleanup
    match result {
        Ok(()) => {
            let duration = start_time.elapsed().unwrap_or_default();
            log_with_timestamp("INFO", &format!("Transcription completed successfully in {:.2} seconds", 
                                               duration.as_secs_f64()));
        }
        Err(error) => {
            log_with_timestamp("ERROR", &format!("Transcription failed: {}", error));
            std::process::exit(1);
        }
    }
    
    log_with_timestamp("INFO", "Enhanced Whisper transcription example finished");
}
