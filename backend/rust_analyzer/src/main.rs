use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

use chrono::NaiveDate;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use csv::ReaderBuilder;

#[derive(Debug, Deserialize)]
struct CombinedDataRow {
    date: String,
    #[serde(flatten)]
    prices_and_returns: HashMap<String, Option<f64>>,
}

#[derive(Debug, Serialize)]
pub struct RegressionResult {
    coin_id: String,
    date: String,
    alpha: Option<f64>,
    beta: Option<f64>,
    // Add confidence intervals
    alpha_lower_90: Option<f64>,
    alpha_upper_90: Option<f64>,
    beta_lower_90: Option<f64>,
    beta_upper_90: Option<f64>,
    window_size: usize,
}

#[derive(Debug, Serialize)]
pub struct AnalysisOutput {
    metadata: AnalysisMetadata,
    results: Vec<RegressionResult>,
}

#[derive(Debug, Serialize)]
pub struct AnalysisMetadata {
    reference_coin: String,
    window_size: usize,
    analysis_date: String,
    total_coins: usize,
}

// Structure to hold regression results with standard errors
struct RegressionOutput {
    alpha: f64,
    beta: f64,
    se_alpha: f64,
    se_beta: f64,
}

pub struct RollingRegressionAnalyzer {
    data: Vec<CombinedDataRow>,
    metadata: serde_json::Value,
    reference_coin: String,
}

impl RollingRegressionAnalyzer {
    pub fn new(data_dir: &str) -> Result<Self, Box<dyn Error>> {
        // Load metadata
        let metadata_path = Path::new(data_dir).join("metadata.json");
        let metadata_file = File::open(&metadata_path)
            .map_err(|e| format!("Failed to open metadata.json: {}", e))?;
        let metadata: serde_json::Value = serde_json::from_reader(metadata_file)?;
        
        let reference_coin = metadata["reference_coin"]
            .as_str()
            .ok_or("Reference coin not found in metadata")?
            .to_string();
        
        println!("Reference coin: {}", reference_coin);
        
        // Load combined data
        let combined_data_path = Path::new(data_dir).join("combined_data.csv");
        let file = File::open(&combined_data_path)
            .map_err(|e| format!("Failed to open combined_data.csv: {}", e))?;
        
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(file);
        
        let mut data = Vec::new();
        let headers = reader.headers()?.clone();
        
        println!("CSV headers: {:?}", headers);
        
        // Read records with better error handling
        for (i, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    let mut row_data = HashMap::new();
                    let mut date = String::new();
                    
                    for (j, field) in record.iter().enumerate() {
                        if j < headers.len() {
                            let header = &headers[j];
                            if header == "date" {
                                date = field.to_string();
                            } else {
                                let value = field.parse::<f64>().ok();
                                row_data.insert(header.to_string(), value);
                            }
                        }
                    }
                    
                    if !date.is_empty() {
                        data.push(CombinedDataRow {
                            date,
                            prices_and_returns: row_data,
                        });
                    }
                }
                Err(e) => {
                    eprintln!("Error reading row {}: {:?}", i + 1, e);
                }
            }
        }
        
        println!("Loaded {} data rows", data.len());
        
        if data.is_empty() {
            return Err("No data rows were loaded".into());
        }
        
        Ok(Self {
            data,
            metadata,
            reference_coin,
        })
    }
    
    /// Compute t-statistic for 90% confidence interval
    fn t_statistic_90(df: usize) -> f64 {
        // For 90% CI, α = 0.10, so we need t(0.05, df)
        // This is a simplified approximation - for production, use a proper stats library
        match df {
            1 => 6.314,
            2 => 2.920,
            3 => 2.353,
            4 => 2.132,
            5 => 2.015,
            6 => 1.943,
            7 => 1.895,
            8 => 1.860,
            9 => 1.833,
            10 => 1.812,
            11..=15 => 1.761,
            16..=20 => 1.729,
            21..=25 => 1.711,
            26..=30 => 1.699,
            31..=40 => 1.684,
            41..=60 => 1.671,
            61..=120 => 1.658,
            _ => 1.645, // For df > 120, approximates normal distribution
        }
    }
    
    /// Perform OLS regression with standard errors and confidence intervals
    fn ols_regression_with_ci(x: &[f64], y: &[f64]) -> Result<RegressionOutput, Box<dyn Error>> {
        if x.len() != y.len() || x.is_empty() {
            return Err("Invalid input dimensions".into());
        }
        
        let n = x.len();
        
        // Create design matrix X = [1, x]
        let mut x_matrix = DMatrix::zeros(n, 2);
        let mut y_vector = DVector::zeros(n);
        
        for i in 0..n {
            x_matrix[(i, 0)] = 1.0;  // Intercept term
            x_matrix[(i, 1)] = x[i];
            y_vector[i] = y[i];
        }
        
        // Calculate coefficients: beta = (X'X)^(-1) X'y
        let xtx = x_matrix.transpose() * &x_matrix;
        let xty = x_matrix.transpose() * &y_vector;
        
        let xtx_inv = xtx.try_inverse()
            .ok_or("Matrix is singular")?;
        
        let coefficients = &xtx_inv * xty;
        
        let alpha = coefficients[0];
        let beta = coefficients[1];
        
        // Calculate fitted values and residuals
        let y_fitted = &x_matrix * &coefficients;
        let residuals = &y_vector - &y_fitted;
        
        // Calculate residual sum of squares
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        
        // Calculate standard errors
        let df = n - 2; // degrees of freedom
        let mse = rss / df as f64; // mean squared error
        
        // Variance-covariance matrix = MSE * (X'X)^(-1)
        let var_cov = xtx_inv * mse;
        
        // Standard errors are square roots of diagonal elements
        let se_alpha = var_cov[(0, 0)].sqrt();
        let se_beta = var_cov[(1, 1)].sqrt();
        
        Ok(RegressionOutput {
            alpha,
            beta,
            se_alpha,
            se_beta,
        })
    }
    
    pub fn analyze_rolling_window(&self, window_size: usize) -> Result<AnalysisOutput, Box<dyn Error>> {
        // Validate window size
        if window_size < 7 || window_size > 180 {
            return Err(format!("Window size must be between 7 and 180 days, got {}", window_size).into());
        }
        
        let mut all_results = Vec::new();
        
        // Get list of coins from metadata
        let coins = self.metadata["coins"]
            .as_array()
            .ok_or("Coins array not found in metadata")?;
        
        println!("Processing {} coins", coins.len());
        
        // For each coin (except reference coin)
        for coin_info in coins {
            let coin_id = coin_info["id"].as_str().ok_or("Coin ID not found")?;
            
            if coin_id == self.reference_coin {
                continue;  // Skip reference coin
            }
            
            println!("Processing coin: {}", coin_id);
            let mut coin_results = Vec::new();
            
            // Perform rolling window regression
            for i in window_size..self.data.len() {
                let window_start = i - window_size;
                
                // Extract returns for this window
                let mut ref_returns = Vec::new();
                let mut coin_returns = Vec::new();
                
                let ref_key = format!("{}_log_return", self.reference_coin);
                let coin_key = format!("{}_log_return", coin_id);
                
                // Collect log returns from window_start to i-1 (inclusive)
                for j in window_start..i {
                    if let (Some(Some(ref_ret)), Some(Some(coin_ret))) = (
                        self.data[j].prices_and_returns.get(&ref_key),
                        self.data[j].prices_and_returns.get(&coin_key)
                    ) {
                        // Only include if both values are finite
                        if ref_ret.is_finite() && coin_ret.is_finite() {
                            ref_returns.push(*ref_ret);
                            coin_returns.push(*coin_ret);
                        }
                    }
                }
                
                // Only perform regression if we have enough valid data points
                if ref_returns.len() >= window_size / 2 {
                    match Self::ols_regression_with_ci(&ref_returns, &coin_returns) {
                        Ok(reg_output) => {
                            // Calculate confidence intervals
                            let df = ref_returns.len() - 2;
                            let t_stat = Self::t_statistic_90(df);
                            
                            let alpha_margin = t_stat * reg_output.se_alpha;
                            let beta_margin = t_stat * reg_output.se_beta;
                            
                            coin_results.push(RegressionResult {
                                coin_id: coin_id.to_string(),
                                date: self.data[i - 1].date.clone(),
                                alpha: Some(reg_output.alpha),
                                beta: Some(reg_output.beta),
                                alpha_lower_90: Some(reg_output.alpha - alpha_margin),
                                alpha_upper_90: Some(reg_output.alpha + alpha_margin),
                                beta_lower_90: Some(reg_output.beta - beta_margin),
                                beta_upper_90: Some(reg_output.beta + beta_margin),
                                window_size,
                            });
                        }
                        Err(_) => {
                            // If regression fails, add null result
                            coin_results.push(RegressionResult {
                                coin_id: coin_id.to_string(),
                                date: self.data[i - 1].date.clone(),
                                alpha: None,
                                beta: None,
                                alpha_lower_90: None,
                                alpha_upper_90: None,
                                beta_lower_90: None,
                                beta_upper_90: None,
                                window_size,
                            });
                        }
                    }
                } else {
                    // Not enough valid data in window
                    coin_results.push(RegressionResult {
                        coin_id: coin_id.to_string(),
                        date: self.data[i - 1].date.clone(),
                        alpha: None,
                        beta: None,
                        alpha_lower_90: None,
                        alpha_upper_90: None,
                        beta_lower_90: None,
                        beta_upper_90: None,
                        window_size,
                    });
                }
            }
            
            println!("  Generated {} results for {}", coin_results.len(), coin_id);
            all_results.extend(coin_results);
        }
        
        let output = AnalysisOutput {
            metadata: AnalysisMetadata {
                reference_coin: self.reference_coin.clone(),
                window_size,
                analysis_date: chrono::Local::now().format("%Y-%m-%d").to_string(),
                total_coins: coins.len() - 1,
            },
            results: all_results,
        };
        
        Ok(output)
    }
    
    pub fn save_results(&self, results: &AnalysisOutput, output_path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::create(output_path)?;
        serde_json::to_writer_pretty(file, results)?;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting Rust regression analyzer with confidence intervals...");
    
    // Try to load analyzer
    let analyzer = match RollingRegressionAnalyzer::new("../data") {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to initialize analyzer: {}", e);
            return Err(e);
        }
    };
    
    // Analyze with different window sizes
    let window_sizes = vec![7, 14, 30, 60, 90, 120, 180];
    
    for window_size in window_sizes {
        println!("\nAnalyzing with window size: {}", window_size);
        
        match analyzer.analyze_rolling_window(window_size) {
            Ok(results) => {
                let output_path = format!("../data/regression_results_window_{}.json", window_size);
                
                match analyzer.save_results(&results, &output_path) {
                    Ok(_) => println!("Saved results to: {}", output_path),
                    Err(e) => eprintln!("Failed to save results: {}", e),
                }
            }
            Err(e) => {
                eprintln!("Failed to analyze window size {}: {}", window_size, e);
            }
        }
    }
    
    println!("\nAnalysis complete with confidence intervals!");
    Ok(())
}