#![allow(non_snake_case)]
extern crate reqwest;
extern crate zip;

use std::io::{Cursor, Read};

use rustlearn::prelude::*;
use rustlearn::feature_extraction::DictVectorizer;
use rustlearn::cross_validation::CrossValidation;
use rustlearn::linear_models::sgdclassifier;
use rustlearn::metrics::accuracy_score;



fn download_file(url: &str) -> Vec<u8> {
	let mut res = reqwest::get(url).unwrap();
	let mut data = Vec::new();
        res.read_to_end(&mut data).unwrap();
	data
}

fn unzip_file(zipped: Vec<u8>) -> String {
	let mut archive = zip::ZipArchive::new(Cursor::new(zipped)).unwrap();
	let mut file = archive.by_name("SMSSpamCollection").unwrap();
	let mut data = String::new();
	file.read_to_string(&mut data).unwrap();

	data
}

fn fit_data(X: &SparseRowArray, y: &Array) -> (f32, f32) {
	let num_epochs = 10;
	let num_folds = 10;

	let mut test_accuracy = 0.0;
	let mut train_accuracy = 0.0;

	for (train_indices, test_indices) in CrossValidation::new(y.rows(), num_folds) {
		let X_train = X.get_rows(&train_indices);
		let X_test = X.get_rows(&test_indices);

		let y_train = y.get_rows(&train_indices);
		let y_test = y.get_rows(&test_indices);

		let mut model = sgdclassifier::Hyperparameters::new(X.cols())
			.learning_rate(0.05)
			.l2_penalty(0.01)
			.build();

		for _ in 0..num_epochs {
			model.fit(&X_train, &y_train).unwrap();
		}
		
		let fold_test_accuracy = accuracy_score(&y_test, &model.predict(&X_test).unwrap());
		let fold_train_accuracy = accuracy_score(&y_train, &model.predict(&X_train).unwrap());

		test_accuracy += fold_test_accuracy;
		train_accuracy += fold_train_accuracy;
	}
	(test_accuracy / num_folds as f32, train_accuracy / num_folds as f32)
}


fn parse_data(data: &str) -> (SparseRowArray, Array) {
	let mut vectorizer = DictVectorizer::new();
	let mut labels = Vec::new();

	for (row_num, line) in data.lines().enumerate() {
		let (label, text) = line.split_at(line.find('\t').unwrap());
		labels.push(match label {
			"spam" => 0.0,
			"ham" => 1.0,
			_ => panic!(format!("Invalid label: {}", label))
		});

		for token in text.split_whitespace() {
			vectorizer.partial_fit(row_num, token, 1.0);
		}
	}
	(vectorizer.transform(), Array::from(labels))
}

fn main() {
    let zipped = download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip");
    println!("Downloaded {} bytes of data", zipped.len());

    let raw_data = unzip_file(zipped);
    for line in raw_data.lines().take(3) {
	println!("{}", line);
    }
   
    let (X, y) = parse_data(&raw_data);
    println!("X: {} rows, {} columns, {} non-zero entries, Y: {:.2}% positive class", X.rows(), X.cols(), X.nnz(), y.mean() * 100.0);

    let start_time = time::precise_time_ns();
    let (test_accuracy, train_accuracy) = fit_data(&X, &y);
    let duration = time::precise_time_ns() - start_time;
 
    println!("Training time: {:.3} seconds", duration as f64 / 1.0e+9);
    println!("Test accuracy: {:.3}, train accuracy: {:.3}", test_accuracy, train_accuracy);
}
