//! # numru-sci
//!
//! A library for advanced scientific computing, built on top of `numru`.
//! Provides tools for optimization, integration, signal processing, and more.
//!
//! ## Usage
//! Add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! sciru = "0.1.0"
//! ```

/// Re-export core functionality for convenience.
pub use numru;

/// Finds the minimum of a univariate function using a simple gradient descent.
///
/// # Arguments
/// * `f` - The function to minimize (takes a single `f64` and returns `f64`).
/// * `x0` - Initial guess for the minimum.
/// * `learning_rate` - Step size for gradient descent.
/// * `max_iter` - Maximum number of iterations.
///
/// # Returns
/// The x-value that minimizes the function.
pub fn minimize_univariate<F>(f: F, x0: f64, learning_rate: f64, max_iter: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut x = x0;
    let epsilon = 1e-6;

    for _ in 0..max_iter {
        let df = (f(x + epsilon) - f(x - epsilon)) / (2.0 * epsilon);
        let step = learning_rate * df;
        x -= step;
        if step.abs() < 1e-8 {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_univariate() {
        let f = |x: f64| x * x;
        let min_x = minimize_univariate(f, 2.0, 0.1, 100);
        assert!((min_x).abs() < 1e-5);
    }
}
