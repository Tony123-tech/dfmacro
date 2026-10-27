use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use regex::Regex;
use chrono::{DateTime, Datelike, Utc, TimeZone};
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (arr, op, scalar=None, other_arr=None))]
fn filter_compare<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    op: &str,
    scalar: Option<f64>,
    other_arr: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let slice = arr.as_slice().unwrap();
    let other_slice = other_arr.map(|a| a.as_slice().unwrap().to_vec());

    let mask: Vec<bool> = py.allow_threads(|| {
        slice.par_iter().enumerate().map(|(i, &v)| {
            if v.is_nan() { return false; }
            let target = if let Some(other) = &other_slice {
                if i < other.len() { other[i] } else { return false; }
            } else if let Some(s) = scalar { s } else { return false; };
            if target.is_nan() { return false; }
            match op {
                ">" => v > target, "<" => v < target, ">=" => v >= target,
                "<=" => v <= target, "==" => (v - target).abs() < f64::EPSILON,
                "!=" => (v - target).abs() >= f64::EPSILON, _ => false,
            }
        }).collect()
    });
    Ok(PyArray1::from_vec_bound(py, mask))
}

#[pyfunction]
#[pyo3(signature = (arr, mask=None))]
fn sum_masked<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>, mask: Option<PyReadonlyArray1<'py, bool>>) -> f64 {
    let slice = arr.as_slice().unwrap();
    let mask_slice = mask.map(|m| m.as_slice().unwrap().to_vec());
    py.allow_threads(|| match mask_slice {
        Some(m) => slice.par_iter().zip(m.par_iter()).filter(|&(&v, &keep)| keep && !v.is_nan()).map(|(&v, _)| v).sum(),
        None => slice.par_iter().filter(|&&v| !v.is_nan()).sum(),
    })
}

#[pyfunction]
#[pyo3(signature = (arr, mask=None))]
fn mean_masked<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>, mask: Option<PyReadonlyArray1<'py, bool>>) -> f64 {
    let slice = arr.as_slice().unwrap();
    let mask_slice = mask.map(|m| m.as_slice().unwrap().to_vec());
    let (sum, count) = py.allow_threads(|| match mask_slice {
        Some(m) => {
            let valid: Vec<f64> = slice.par_iter().zip(m.par_iter()).filter(|&(&v, &keep)| keep && !v.is_nan()).map(|(&v, _)| v).collect();
            (valid.par_iter().sum::<f64>(), valid.len())
        }
        None => {
            let valid: Vec<f64> = slice.par_iter().filter(|&&v| !v.is_nan()).copied().collect();
            (valid.par_iter().sum::<f64>(), valid.len())
        }
    });
    if count == 0 { f64::NAN } else { sum / (count as f64) }
}

#[pyfunction]
fn var_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>) -> f64 {
    let slice = arr.as_slice().unwrap();
    let (mean, count) = py.allow_threads(|| {
        let valid: Vec<f64> = slice.par_iter().filter(|&&v| !v.is_nan()).copied().collect();
        (valid.par_iter().sum::<f64>() / valid.len() as f64, valid.len())
    });
    if count <= 1 { return f64::NAN; }
    py.allow_threads(|| {
        let sum_sq_diff: f64 = slice.par_iter().filter(|&&v| !v.is_nan()).map(|&v| (v - mean).powi(2)).sum();
        sum_sq_diff / (count - 1) as f64
    })
}

#[pyfunction]
fn std_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>) -> f64 {
    var_parallel(py, arr).sqrt()
}

#[pyfunction]
fn corr_parallel<'py>(py: Python<'py>, arr1: PyReadonlyArray1<'py, f64>, arr2: PyReadonlyArray1<'py, f64>) -> f64 {
    let s1 = arr1.as_slice().unwrap();
    let s2 = arr2.as_slice().unwrap();
    if s1.len() != s2.len() { return f64::NAN; }
    let (mean1, mean2) = py.allow_threads(|| {
        let m1 = s1.par_iter().zip(s2.par_iter()).filter(|&(&a, &b)| !a.is_nan() && !b.is_nan()).map(|(&a, _)| a).sum::<f64>();
        let m2 = s1.par_iter().zip(s2.par_iter()).filter(|&(&a, &b)| !a.is_nan() && !b.is_nan()).map(|(_, &b)| b).sum::<f64>();
        let n = s1.par_iter().zip(s2.par_iter()).filter(|&(&a, &b)| !a.is_nan() && !b.is_nan()).count() as f64;
        (m1 / n, m2 / n)
    });
    py.allow_threads(|| {
        let mut num = 0.0;
        let mut d1 = 0.0;
        let mut d2 = 0.0;
        for (&a, &b) in s1.iter().zip(s2.iter()) {
            if !a.is_nan() && !b.is_nan() {
                let da = a - mean1;
                let db = b - mean2;
                num += da * db;
                d1 += da * da;
                d2 += db * db;
            }
        }
        num / (d1 * d2).sqrt()
    })
}

#[pyfunction]
fn cov_parallel<'py>(py: Python<'py>, arr1: PyReadonlyArray1<'py, f64>, arr2: PyReadonlyArray1<'py, f64>) -> f64 {
    let s1 = arr1.as_slice().unwrap();
    let s2 = arr2.as_slice().unwrap();
    if s1.len() != s2.len() { return f64::NAN; }
    let (mean1, mean2, n) = py.allow_threads(|| {
        let m1 = s1.par_iter().zip(s2.par_iter()).filter(|&(&a, &b)| !a.is_nan() && !b.is_nan()).map(|(&a, _)| a).sum::<f64>();
        let m2 = s1.par_iter().zip(s2.par_iter()).filter(|&(&a, &b)| !a.is_nan() && !b.is_nan()).map(|(_, &b)| b).sum::<f64>();
        let n = s1.par_iter().zip(s2.par_iter()).filter(|&(&a, &b)| !a.is_nan() && !b.is_nan()).count() as f64;
        (m1 / n, m2 / n, n)
    });
    if n <= 1.0 { return f64::NAN; }
    py.allow_threads(|| {
        let mut sum = 0.0;
        for (&a, &b) in s1.iter().zip(s2.iter()) {
            if !a.is_nan() && !b.is_nan() {
                sum += (a - mean1) * (b - mean2);
            }
        }
        sum / (n - 1.0)
    })
}

#[pyfunction]
fn cumsum_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut result = Vec::with_capacity(slice.len());
    let mut current = 0.0;
    for &v in slice {
        if !v.is_nan() { current += v; }
        result.push(current);
    }
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn cummax_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut result = Vec::with_capacity(slice.len());
    let mut current = f64::NEG_INFINITY;
    for &v in slice {
        if !v.is_nan() && v > current { current = v; }
        result.push(current);
    }
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn cummin_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut result = Vec::with_capacity(slice.len());
    let mut current = f64::INFINITY;
    for &v in slice {
        if !v.is_nan() && v < current { current = v; }
        result.push(current);
    }
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn math_scalar<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>, op: &str, scalar: f64) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let result: Vec<f64> = py.allow_threads(|| {
        slice.par_iter().map(|&v| match op {
            "+" => v + scalar, "-" => v - scalar, "*" => v * scalar, "/" => v / scalar, _ => v,
        }).collect()
    });
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn sort_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>, descending: bool) -> Bound<'py, PyArray1<f64>> {
    let mut slice = arr.as_slice().unwrap().to_vec();
    py.allow_threads(|| {
        slice.par_sort_unstable_by(|a, b| {
            if descending { b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal) }
            else { a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) }
        });
    });
    PyArray1::from_vec_bound(py, slice)
}

#[pyfunction]
fn nlargest_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>, n: usize) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut vec = slice.to_vec();
    py.allow_threads(|| {
        vec.par_sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        vec.truncate(n);
    });
    PyArray1::from_vec_bound(py, vec)
}

#[pyfunction]
fn nsmallest_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>, n: usize) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut vec = slice.to_vec();
    py.allow_threads(|| {
        vec.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vec.truncate(n);
    });
    PyArray1::from_vec_bound(py, vec)
}

#[pyfunction]
fn unique_parallel(py: Python<'_>, arr: Vec<f64>) -> PyResult<Py<PyAny>> {
    let result: Vec<f64> = py.allow_threads(|| {
        use std::collections::HashSet;
        let set: HashSet<u64> = arr.into_iter().map(|v| v.to_bits()).collect();
        set.into_iter().map(f64::from_bits).collect()
    });
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn value_counts_parallel(py: Python<'_>, arr: Vec<f64>) -> PyResult<Py<PyAny>> {
    let map: HashMap<u64, usize> = py.allow_threads(|| {
        let mut m = HashMap::new();
        for v in arr {
            *m.entry(v.to_bits()).or_insert(0) += 1;
        }
        m
    });

    let mut result: Vec<(f64, usize)> = map.into_iter().map(|(bits, count)| (f64::from_bits(bits), count)).collect();
    result.sort_by(|a, b| b.1.cmp(&a.1));
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn text_lower_parallel(py: Python<'_>, arr: Vec<String>) -> PyResult<Py<PyAny>> {
    let result: Vec<String> = py.allow_threads(|| arr.into_par_iter().map(|s| s.to_lowercase()).collect());
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn text_upper_parallel(py: Python<'_>, arr: Vec<String>) -> PyResult<Py<PyAny>> {
    let result: Vec<String> = py.allow_threads(|| arr.into_par_iter().map(|s| s.to_uppercase()).collect());
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn text_contains_parallel(py: Python<'_>, arr: Vec<String>, pat: String) -> PyResult<Py<PyAny>> {
    let re = Regex::new(&pat).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result: Vec<bool> = py.allow_threads(|| arr.into_par_iter().map(|s| re.is_match(&s)).collect());
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn text_len_parallel(py: Python<'_>, arr: Vec<String>) -> PyResult<Py<PyAny>> {
    let result: Vec<i64> = py.allow_threads(|| arr.into_par_iter().map(|s| s.len() as i64).collect());
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn text_replace_parallel(py: Python<'_>, arr: Vec<String>, pat: String, repl: String) -> PyResult<Py<PyAny>> {
    let re = Regex::new(&pat).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result: Vec<String> = py.allow_threads(|| arr.into_par_iter().map(|s| re.replace_all(&s, repl.as_str()).to_string()).collect());
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn text_strip_parallel(py: Python<'_>, arr: Vec<String>) -> PyResult<Py<PyAny>> {
    let result: Vec<String> = py.allow_threads(|| arr.into_par_iter().map(|s| s.trim().to_string()).collect());
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn groupby_sum_parallel<'py>(py: Python<'py>, keys: Vec<String>, values: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vals = values.as_slice().unwrap();
    if keys.len() != vals.len() { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Length mismatch")); }

    let map: HashMap<String, f64> = py.allow_threads(|| {
        let mut m = HashMap::new();
        for (k, &v) in keys.iter().zip(vals.iter()) {
            if !v.is_nan() {
                *m.entry(k.clone()).or_insert(0.0) += v;
            }
        }
        m
    });

    let mut result: Vec<f64> = map.into_values().collect();
    result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
fn parse_datetime_parallel(py: Python<'_>, arr: Vec<String>, fmt: &str) -> PyResult<Py<PyAny>> {
    let result: Vec<i64> = py.allow_threads(|| {
        arr.into_par_iter().filter_map(|s| {
            DateTime::parse_from_str(&s, fmt).ok().map(|dt| dt.timestamp())
        }).collect()
    });
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn dt_year_parallel(py: Python<'_>, arr: Vec<i64>) -> PyResult<Py<PyAny>> {
    let result: Vec<i32> = py.allow_threads(|| {
        arr.into_par_iter().filter_map(|ts| {
            Utc.timestamp_opt(ts, 0).single().map(|dt| dt.year())
        }).collect()
    });
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn dt_month_parallel(py: Python<'_>, arr: Vec<i64>) -> PyResult<Py<PyAny>> {
    let result: Vec<u32> = py.allow_threads(|| {
        arr.into_par_iter().filter_map(|ts| {
            Utc.timestamp_opt(ts, 0).single().map(|dt| dt.month())
        }).collect()
    });
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn dt_day_parallel(py: Python<'_>, arr: Vec<i64>) -> PyResult<Py<PyAny>> {
    let result: Vec<u32> = py.allow_threads(|| {
        arr.into_par_iter().filter_map(|ts| {
            Utc.timestamp_opt(ts, 0).single().map(|dt| dt.day())
        }).collect()
    });
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pymodule]
fn dfmacro_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_compare, m)?)?;
    m.add_function(wrap_pyfunction!(sum_masked, m)?)?;
    m.add_function(wrap_pyfunction!(mean_masked, m)?)?;
    m.add_function(wrap_pyfunction!(var_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(std_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(corr_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(cov_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(cummax_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(cummin_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(math_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(sort_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(nlargest_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(nsmallest_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(unique_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(value_counts_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_lower_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_upper_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_contains_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_len_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_replace_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_strip_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_sum_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(parse_datetime_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(dt_year_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(dt_month_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(dt_day_parallel, m)?)?;
    Ok(())
}
