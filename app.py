import os
import logging
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from prediction import load_model_and_scaler, make_prediction, process_batch_predictions, get_feature_names

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-dev")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Load model and scaler
model, scaler = load_model_and_scaler()
feature_names = get_feature_names()

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {}
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                flash(f"Missing value for {feature}", "danger")
                return redirect(url_for('index'))
            try:
                data[feature] = float(value)
            except ValueError:
                flash(f"Invalid value for {feature}. Please enter a number.", "danger")
                return redirect(url_for('index'))
        
        # Make prediction
        result, probability, explanation = make_prediction(model, scaler, data)
        
        # Store result in session for visualization
        session['prediction_result'] = {
            'result': int(result),
            'probability': probability,
            'features': data,
            'explanation': explanation
        }
        
        return render_template('result.html', 
                              result=result, 
                              probability=probability,
                              features=data,
                              explanation=explanation)
    
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the CSV file
            results_df, summary = process_batch_predictions(model, scaler, filepath)
            
            # Store results in session
            session['batch_results'] = results_df.to_dict('records')
            session['batch_summary'] = summary
            
            # Clean up the file
            os.remove(filepath)
            
            return render_template('batch_result.html', 
                                  results=results_df.to_dict('records'),
                                  summary=summary)
        
        except Exception as e:
            logging.error(f"Error in batch prediction: {str(e)}")
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('index'))
    else:
        flash('Allowed file type is CSV', 'danger')
        return redirect(url_for('index'))

@app.route('/get_prediction_data')
def get_prediction_data():
    """Endpoint to get prediction data for charts"""
    prediction_result = session.get('prediction_result', {})
    return jsonify(prediction_result)

@app.route('/get_batch_data')
def get_batch_data():
    """Endpoint to get batch prediction data for charts"""
    batch_results = session.get('batch_results', [])
    batch_summary = session.get('batch_summary', {})
    return jsonify({
        'results': batch_results,
        'summary': batch_summary
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
