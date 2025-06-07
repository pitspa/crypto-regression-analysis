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
    prices_and_returns: HashMap<String, f64>,
}

#[derive(Debug, Serialize)]
struct RegressionResult {
    coin_id: String,
    date: String,
    alpha: Option<f64>,
    beta: Option<f64>,
    window_size: usize,
}

#[derive(Debug, Serialize)]
struct AnalysisOutput {
    metadata: AnalysisMetadata,
    results: Vec<RegressionResult>,
}

#[derive(Debug, Serialize)]
struct AnalysisMetadata {
    reference_coin: String,
    window_size: usize,
    analysis_date: String,
    total_coins: usize,
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
        let metadata_file = File::open(metadata_path)?;
        let metadata: serde_json::Value = serde_json::from_reader(metadata_file)?;
        
        let reference_coin = metadata["reference_coin"]
            .as_str()
            .ok_or("Reference coin not found in metadata")?
            .to_string();
        
        // Load combined data
        let combined_data_path = Path::new(data_dir).join("combined_data.csv");
        let file = File::open(combined_data_path)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        
        let mut data = Vec::new();
        for result in reader.deserialize() {
            let row: CombinedDataRow = result?;
            data.push(row);
        }
        
        Ok(Self {
            data,
            metadata,
            reference_coin,
        })
    }
    
    /// Perform OLS regression: y = alpha + beta * x + epsilon
    fn ols_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64), Box<dyn Error>> {
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
        
        let coefficients = xtx.try_inverse()
            .ok_or("Matrix is singular")?
            * xty;
        
        let alpha = coefficients[0];
        let beta = coefficients[1];
        
        Ok((alpha, beta))
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
        
        // For each coin (except reference coin)
        for coin_info in coins {
            let coin_id = coin_info["id"].as_str().ok_or("Coin ID not found")?;
            
            if coin_id == self.reference_coin {
                continue;  // Skip reference coin
            }
            
            let mut coin_results = Vec::new();
            
            // Perform rolling window regression
            // For a window of size W at position i:
            // - We use log returns from positions [i-W, i-1] (inclusive)
            // - This gives us W log returns
            // - The result is associated with date at position i-1
            for i in window_size..self.data.len() {
                let window_start = i - window_size;
                
                // Extract returns for this window
                let mut ref_returns = Vec::new();
                let mut coin_returns = Vec::new();
                
                let ref_key = format!("{}_log_return", self.reference_coin);
                let coin_key = format!("{}_log_return", coin_id);
                
                // Collect log returns from window_start to i-1 (inclusive)
                for j in window_start..i {
                    if let (Some(&ref_ret), Some(&coin_ret)) = (
                        self.data[j].prices_and_returns.get(&ref_key),
                        self.data[j].prices_and_returns.get(&coin_key)
                    ) {
                        // Only include if both values are finite (not NaN or infinity)
                        if ref_ret.is_finite() && coin_ret.is_finite() {
                            ref_returns.push(ref_ret);
                            coin_returns.push(coin_ret);
                        }
                    }
                }
                
                // Only perform regression if we have enough valid data points
                // Require at least 50% of the window size to be valid
                if ref_returns.len() >= window_size / 2 {
                    match Self::ols_regression(&ref_returns, &coin_returns) {
                        Ok((alpha, beta)) => {
                            coin_results.push(RegressionResult {
                                coin_id: coin_id.to_string(),
                                date: self.data[i - 1].date.clone(),
                                alpha: Some(alpha),
                                beta: Some(beta),
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
                        window_size,
                    });
                }
            }
            
            all_results.extend(coin_results);
        }
        
        let output = AnalysisOutput {
            metadata: AnalysisMetadata {
                reference_coin: self.reference_coin.clone(),
                window_size,
                analysis_date: chrono::Local::now().format("%Y-%m-%d").to_string(),
                total_coins: coins.len() - 1,  // Excluding reference coin
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ols_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let (alpha, beta) = RollingRegressionAnalyzer::ols_regression(&x, &y).unwrap();
        
        assert!((alpha - 0.0).abs() < 0.001);
        assert!((beta - 2.0).abs() < 0.001);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Example usage
    let analyzer = RollingRegressionAnalyzer::new("../data")?;
    
    // Analyze with different window sizes (constrained between 7 and 180 days)
    let window_sizes = vec![7, 14, 30, 60, 90, 120, 180];
    
    for window_size in window_sizes {
        if window_size < 7 || window_size > 180 {
            println!("Skipping invalid window size: {}", window_size);
            continue;
        }
        
        println!("Analyzing with window size: {}", window_size);
        
        let results = analyzer.analyze_rolling_window(window_size)?;
        let output_path = format!("../data/regression_results_window_{}.json", window_size);
        
        analyzer.save_results(&results, &output_path)?;
        println!("Saved results to: {}", output_path);
    }
    
    Ok(())
}