/**
 * Enhanced Chinese Speech Recognition Example for Whisper-RS
 * 
 * This example demonstrates Chinese speech-to-text transcription capabilities using whisper-rs.
 * It loads a Whisper model and processes Chinese audio files to generate timestamped transcriptions.
 * 
 * Features:
 * - Chinese language support with proper encoding
 * - Model loading with proper error handling
 * - Audio format conversion and validation
 * - Transcription with segment timestamps
 * - Comprehensive logging with timestamps
 * - Enhanced error handling and validation
 * - Unicode text processing for Chinese characters
 * 
 * Usage:
 *   cargo run --example chinese_test ggml-small.bin chinese_audio.wav
 *   cargo run --example chinese_test ggml-medium.bin chinese_speech.wav
 * 
 * Dependencies:
 *   - whisper-rs: Main library for Whisper functionality
 *   - hound: WAV file reading and processing
 * 
 * Author: Enhanced for whisper-rs Chinese recognition
 * Date: 2024
 * License: Same as whisper-rs project
 */

use std::time::SystemTime;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/**
 * Print a timestamped log message to stdout with Chinese support
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
 * Validate command line arguments for Chinese speech recognition
 * 
 * @return Result containing (model_path, wav_path) or error message
 */
fn validate_arguments() -> Result<(String, String), String> {
    let _timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", "验证命令行参数 (Validating command line arguments)");
    
    let model_path = std::env::args()
        .nth(1)
        .ok_or_else(|| {
            log_with_timestamp("ERROR", "缺少模型路径参数 (Missing model path argument)");
            "请指定模型文件路径作为第一个参数 (Please specify model file path as first argument)".to_string()
        })?;
    
    let wav_path = std::env::args()
        .nth(2)
        .ok_or_else(|| {
            log_with_timestamp("ERROR", "缺少WAV文件路径参数 (Missing WAV file path argument)");
            "请指定WAV文件路径作为第二个参数 (Please specify WAV file path as second argument)".to_string()
        })?;
    
    // Validate file existence
    if !std::path::Path::new(&model_path).exists() {
        let error_msg = format!("模型文件不存在 (Model file does not exist): {}", model_path);
        log_with_timestamp("ERROR", &error_msg);
        return Err(error_msg);
    }
    
    if !std::path::Path::new(&wav_path).exists() {
        let error_msg = format!("WAV文件不存在 (WAV file does not exist): {}", wav_path);
        log_with_timestamp("ERROR", &error_msg);
        return Err(error_msg);
    }
    
    log_with_timestamp("INFO", &format!("模型路径 (Model path): {}", model_path));
    log_with_timestamp("INFO", &format!("音频路径 (Audio path): {}", wav_path));
    
    Ok((model_path, wav_path))
}

/**
 * Load and validate Chinese audio WAV file
 * 
 * @param wav_path - Path to the WAV file
 * @return Result containing audio samples or error message
 */
fn load_chinese_audio_samples(wav_path: &str) -> Result<Vec<i16>, String> {
    let _timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", &format!("加载中文音频文件 (Loading Chinese audio file): {}", wav_path));
    
    let wav_reader = hound::WavReader::open(wav_path)
        .map_err(|e| {
            let error_msg = format!("无法打开WAV文件 (Failed to open WAV file): {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    let spec = wav_reader.spec();
    log_with_timestamp("INFO", &format!("音频规格 (Audio specs) - 采样率 (Sample rate): {}Hz, 声道 (Channels): {}, 位深 (Bits per sample): {}", 
                                       spec.sample_rate, spec.channels, spec.bits_per_sample));
    
    let samples: Result<Vec<i16>, _> = wav_reader
        .into_samples::<i16>()
        .collect();
    
    let samples = samples.map_err(|e| {
        let error_msg = format!("无法读取音频样本 (Failed to read audio samples): {}", e);
        log_with_timestamp("ERROR", &error_msg);
        error_msg
    })?;
    
    log_with_timestamp("INFO", &format!("成功加载音频样本 (Successfully loaded audio samples): {}", samples.len()));
    
    if samples.is_empty() {
        let error_msg = "音频文件不包含样本数据 (Audio file contains no samples)";
        log_with_timestamp("ERROR", error_msg);
        return Err(error_msg.to_string());
    }
    
    Ok(samples)
}

/**
 * Initialize Whisper context and state for Chinese recognition
 * 
 * @param model_path - Path to the Whisper model file
 * @return Result containing (context, state) or error message
 */
fn initialize_chinese_whisper_context(model_path: &str) -> Result<(WhisperContext, whisper_rs::WhisperState), String> {
    let _timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", &format!("加载中文Whisper模型 (Loading Chinese Whisper model): {}", model_path));
    
    let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .map_err(|e| {
            let error_msg = format!("无法加载模型 (Failed to load model): {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    log_with_timestamp("INFO", "中文Whisper模型加载成功 (Chinese Whisper model loaded successfully)");
    
    let state = ctx.create_state().map_err(|e| {
        let error_msg = format!("无法创建Whisper状态 (Failed to create Whisper state): {}", e);
        log_with_timestamp("ERROR", &error_msg);
        error_msg
    })?;
    
    log_with_timestamp("INFO", "中文Whisper状态创建成功 (Chinese Whisper state created successfully)");
    
    Ok((ctx, state))
}

/**
 * Configure Whisper parameters for Chinese transcription
 * 
 * @param language - Target language for transcription (zh for Chinese)
 * @return Configured FullParams for Whisper
 */
fn configure_chinese_whisper_params(language: &str) -> FullParams {
    let _timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", &format!("配置中文Whisper参数 (Configuring Chinese Whisper parameters) for language: {}", language));
    
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    
    // Set the language for Chinese transcription
    params.set_language(Some(language));
    
    // Enable translation if needed (to translate to English)
    // params.set_translate(true);  // Uncomment if you want translation to English
    
    // Disable console output to keep our logs clean
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    
    // Set temperature for better Chinese recognition
    params.set_temperature(0.0);
    
    log_with_timestamp("INFO", "中文Whisper参数配置成功 (Chinese Whisper parameters configured successfully)");
    
    params
}

/**
 * Process audio samples for Chinese Whisper model compatibility
 * 
 * @param samples - Raw integer audio samples
 * @return Result containing processed f32 mono samples or error message
 */
fn process_chinese_audio_samples(samples: &[i16]) -> Result<Vec<f32>, String> {
    let _timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", "转换中文音频样本以兼容Whisper (Converting Chinese audio samples for Whisper compatibility)");
    
    // Convert integer samples to float
    let mut inter_samples = vec![0.0f32; samples.len()];
    
    whisper_rs::convert_integer_to_float_audio(samples, &mut inter_samples)
        .map_err(|e| {
            let error_msg = format!("无法将整数转换为浮点音频 (Failed to convert integer to float audio): {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    log_with_timestamp("INFO", "整数到浮点转换完成 (Integer to float conversion completed)");
    
    // Convert stereo to mono if needed
    let mono_samples = whisper_rs::convert_stereo_to_mono_audio(&inter_samples)
        .map_err(|e| {
            let error_msg = format!("无法将立体声转换为单声道音频 (Failed to convert stereo to mono audio): {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    log_with_timestamp("INFO", &format!("中文音频处理完成 (Chinese audio processing completed). 最终样本数 (Final sample count): {}", mono_samples.len()));
    
    Ok(mono_samples)
}

/**
 * Run Chinese transcription and extract results
 * 
 * @param state - Whisper state for processing
 * @param params - Configured Whisper parameters
 * @param samples - Processed audio samples
 * @return Result indicating success or error message
 */
fn run_chinese_transcription(
    state: &mut whisper_rs::WhisperState,
    params: FullParams,
    samples: &[f32]
) -> Result<(), String> {
    let _timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", "开始中文转录过程 (Starting Chinese transcription process)");
    
    // Run the Whisper model
    state.full(params, samples).map_err(|e| {
        let error_msg = format!("无法运行Whisper模型 (Failed to run Whisper model): {}", e);
        log_with_timestamp("ERROR", &error_msg);
        error_msg
    })?;
    
    log_with_timestamp("INFO", "中文转录成功完成 (Chinese transcription completed successfully)");
    
    // Extract and display results
    let num_segments = state.full_n_segments().map_err(|e| {
        let error_msg = format!("无法获取段落数量 (Failed to get number of segments): {}", e);
        log_with_timestamp("ERROR", &error_msg);
        error_msg
    })?;
    
    log_with_timestamp("INFO", &format!("找到中文转录段落 (Found Chinese transcription segments): {}", num_segments));
    
    if num_segments == 0 {
        log_with_timestamp("WARN", "未找到转录段落 (No transcription segments found)");
        return Ok(());
    }
    
    println!("\n=== 中文转录结果 (CHINESE TRANSCRIPTION RESULTS) ===");
    
    for i in 0..num_segments {
        let segment = state.full_get_segment_text(i).map_err(|e| {
            let error_msg = format!("无法获取段落文本 (Failed to get segment text) for segment {}: {}", i, e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
        
        let start_timestamp = state.full_get_segment_t0(i).map_err(|e| {
            let error_msg = format!("无法获取开始时间戳 (Failed to get start timestamp) for segment {}: {}", i, e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
        
        let end_timestamp = state.full_get_segment_t1(i).map_err(|e| {
            let error_msg = format!("无法获取结束时间戳 (Failed to get end timestamp) for segment {}: {}", i, e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
        
        // Check if this looks like Chinese text (contains Chinese characters)
        let contains_chinese = segment.chars().any(|c| {
            matches!(c, '\u{4e00}'..='\u{9fff}' | '\u{3400}'..='\u{4dbf}' | '\u{20000}'..='\u{2a6df}')
        });
        
        let language_indicator = if contains_chinese { "🇨🇳" } else { "🇺🇸" };
        
        println!("[{} - {}] {}: {}", start_timestamp, end_timestamp, language_indicator, segment);
    }
    
    println!("=== 中文转录结果结束 (END CHINESE TRANSCRIPTION RESULTS) ===\n");
    
    Ok(())
}

/**
 * Main function - Enhanced Chinese Whisper transcription example
 * 
 * Processes command line arguments, loads audio and model files,
 * performs Chinese transcription with comprehensive error handling and logging.
 */
fn main() {
    let start_time = SystemTime::now();
    log_with_timestamp("INFO", "开始中文语音识别示例 (Starting enhanced Chinese speech recognition example)");
    
    // Process execution with comprehensive error handling
    let result = (|| -> Result<(), String> {
        // Validate command line arguments
        let (model_path, wav_path) = validate_arguments()?;
        
        // Load audio samples
        let samples = load_chinese_audio_samples(&wav_path)?;
        
        // Initialize Whisper context and state
        let (_ctx, mut state) = initialize_chinese_whisper_context(&model_path)?;
        
        // Configure transcription parameters for Chinese
        let language = "zh"; // Chinese language code
        let params = configure_chinese_whisper_params(language);
        
        // Process audio samples for Whisper compatibility
        let processed_samples = process_chinese_audio_samples(&samples)?;
        
        // Run Chinese transcription and display results
        run_chinese_transcription(&mut state, params, &processed_samples)?;
        
        Ok(())
    })();
    
    // Handle final result and cleanup
    match result {
        Ok(()) => {
            let duration = start_time.elapsed().unwrap_or_default();
            log_with_timestamp("INFO", &format!("中文转录成功完成 (Chinese transcription completed successfully) in {:.2} seconds", 
                                               duration.as_secs_f64()));
        }
        Err(error) => {
            log_with_timestamp("ERROR", &format!("中文转录失败 (Chinese transcription failed): {}", error));
            std::process::exit(1);
        }
    }
    
    log_with_timestamp("INFO", "中文语音识别示例结束 (Enhanced Chinese speech recognition example finished)");
} 