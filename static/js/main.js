// Main JavaScript for Bank Churn Prediction App

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });

    // Input validation - numeric inputs only
    const numericInputs = document.querySelectorAll('input[type="number"]');
    
    numericInputs.forEach(input => {
        input.addEventListener('input', function() {
            // Allow only numbers and decimal point
            this.value = this.value.replace(/[^0-9.]/g, '');
            
            // Ensure only one decimal point
            const decimalCount = (this.value.match(/\./g) || []).length;
            if (decimalCount > 1) {
                this.value = this.value.replace(/\.(?=.*\.)/, '');
            }
        });
    });

    // Geography dropdown value handling
    const geographySelect = document.getElementById('Geography');
    if (geographySelect) {
        geographySelect.addEventListener('change', function() {
            // Store the encoded value in a hidden input
            const geographyValue = document.getElementById('geography-value');
            switch(this.value) {
                case 'France':
                    geographyValue.value = '0';
                    break;
                case 'Germany':
                    geographyValue.value = '1';
                    break;
                case 'Spain':
                    geographyValue.value = '2';
                    break;
                default:
                    geographyValue.value = '';
            }
        });
    }

    // Gender dropdown value handling
    const genderSelect = document.getElementById('Gender');
    if (genderSelect) {
        genderSelect.addEventListener('change', function() {
            // Store the encoded value in a hidden input
            const genderValue = document.getElementById('gender-value');
            switch(this.value) {
                case 'Female':
                    genderValue.value = '0';
                    break;
                case 'Male':
                    genderValue.value = '1';
                    break;
                default:
                    genderValue.value = '';
            }
        });
    }

    // File input validation
    const fileInput = document.getElementById('csv-file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileFeedback = document.getElementById('file-feedback');
            const submitBtn = document.getElementById('batch-submit-btn');
            
            if (this.files.length === 0) {
                fileFeedback.textContent = 'Please select a file';
                fileFeedback.className = 'invalid-feedback';
                submitBtn.disabled = true;
                return;
            }
            
            const file = this.files[0];
            const fileName = file.name;
            const fileExtension = fileName.split('.').pop().toLowerCase();
            
            if (fileExtension !== 'csv') {
                fileFeedback.textContent = 'Only CSV files are allowed';
                fileFeedback.className = 'invalid-feedback d-block';
                submitBtn.disabled = true;
                return;
            }
            
            // Valid file
            fileFeedback.textContent = `Selected file: ${fileName}`;
            fileFeedback.className = 'valid-feedback d-block';
            submitBtn.disabled = false;
            
            // Update file name display
            const fileNameDisplay = document.getElementById('file-name-display');
            if (fileNameDisplay) {
                fileNameDisplay.textContent = fileName;
            }
        });
    }

    // Toggle between individual and batch prediction
    const predictionTabs = document.querySelectorAll('.prediction-tab');
    if (predictionTabs.length > 0) {
        predictionTabs.forEach(tab => {
            tab.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Remove active class from all tabs
                predictionTabs.forEach(t => {
                    t.classList.remove('active');
                    t.setAttribute('aria-selected', 'false');
                });
                
                // Add active class to clicked tab
                this.classList.add('active');
                this.setAttribute('aria-selected', 'true');
                
                // Show corresponding content
                const target = this.getAttribute('data-bs-target');
                document.querySelectorAll('.tab-pane').forEach(pane => {
                    pane.classList.remove('show', 'active');
                });
                document.querySelector(target).classList.add('show', 'active');
            });
        });
    }

    // Feature engineering calculations
    const calculateDerivedFeatures = () => {
        const balance = parseFloat(document.getElementById('Balance').value) || 0;
        const salary = parseFloat(document.getElementById('EstimatedSalary').value) || 1;
        const age = parseFloat(document.getElementById('Age').value) || 1;
        const tenure = parseFloat(document.getElementById('Tenure').value) || 0;
        const creditScore = parseFloat(document.getElementById('CreditScore').value) || 0;
        
        // Calculate derived features
        const balanceSalaryRatio = balance / salary;
        const tenureByAge = tenure / age;
        const creditScorePerAge = creditScore / age;
        
        // Update hidden fields
        document.getElementById('BalanceSalaryRatio').value = balanceSalaryRatio.toFixed(4);
        document.getElementById('TenureByAge').value = tenureByAge.toFixed(4);
        document.getElementById('CreditScorePerAge').value = creditScorePerAge.toFixed(4);
    };
    
    // Add event listeners to calculate derived features
    ['Balance', 'EstimatedSalary', 'Age', 'Tenure', 'CreditScore'].forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', calculateDerivedFeatures);
            element.addEventListener('input', calculateDerivedFeatures);
        }
    });

    // Sample customer profiles
    const sampleProfiles = document.querySelectorAll('.sample-profile');
    if (sampleProfiles.length > 0) {
        sampleProfiles.forEach(profile => {
            profile.addEventListener('click', function(e) {
                e.preventDefault();
                
                const profileType = this.getAttribute('data-profile');
                let profileData = {};
                
                switch(profileType) {
                    case 'low-risk':
                        profileData = {
                            CreditScore: 850,  // Excellent credit score
                            Geography: 'France',
                            GeographyValue: 0,
                            Gender: 'Female',
                            GenderValue: 0,
                            Age: 45,  // Middle-aged (stable)
                            Tenure: 10, // Long-term customer
                            Balance: 175000, // High balance
                            NumOfProducts: 2, // Optimal number of products
                            HasCrCard: 1,
                            IsActiveMember: 1, // Active member
                            EstimatedSalary: 95000 // Good salary
                        };
                        break;
                    case 'medium-risk':
                        profileData = {
                            CreditScore: 650, // Average credit score
                            Geography: 'Germany',
                            GeographyValue: 1,
                            Gender: 'Male',
                            GenderValue: 1,
                            Age: 35,
                            Tenure: 3, // Newer customer
                            Balance: 40000, // Moderate balance
                            NumOfProducts: 1, // Only one product
                            HasCrCard: 1,
                            IsActiveMember: 0, // Not active
                            EstimatedSalary: 60000
                        };
                        break;
                    case 'high-risk':
                        profileData = {
                            CreditScore: 480, // Poor credit score
                            Geography: 'Spain',
                            GeographyValue: 2,
                            Gender: 'Male',
                            GenderValue: 1,
                            Age: 25, // Young customer
                            Tenure: 1, // New customer
                            Balance: 0, // Zero balance
                            NumOfProducts: 4, // Too many products
                            HasCrCard: 0, // No credit card
                            IsActiveMember: 0, // Inactive
                            EstimatedSalary: 50000 // Lower salary
                        };
                        break;
                }
                
                // Fill the form with sample data
                Object.keys(profileData).forEach(key => {
                    const element = document.getElementById(key);
                    if (element) {
                        element.value = profileData[key];
                    }
                });
                
                // Select dropdowns
                document.getElementById('Geography').value = profileData.Geography;
                document.getElementById('Gender').value = profileData.Gender;
                document.getElementById('geography-value').value = profileData.GeographyValue;
                document.getElementById('gender-value').value = profileData.GenderValue;
                
                // Calculate derived features
                calculateDerivedFeatures();
            });
        });
    }
});
