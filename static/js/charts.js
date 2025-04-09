// Charts for visualizing prediction results

// Initialize individual prediction chart
function initIndividualPredictionChart() {
    // Fetch prediction data from server
    fetch('/get_prediction_data')
        .then(response => response.json())
        .then(data => {
            if (!data || Object.keys(data).length === 0) {
                console.log('No prediction data available');
                return;
            }
            
            // Create probability gauge chart
            createProbabilityGauge(data.probability);
            
            // Create feature importance chart
            createFeatureImportanceChart(data.features);
        })
        .catch(error => console.error('Error fetching prediction data:', error));
}

// Initialize batch prediction charts
function initBatchPredictionCharts() {
    // Fetch batch prediction data from server
    fetch('/get_batch_data')
        .then(response => response.json())
        .then(data => {
            if (!data || !data.summary || Object.keys(data.summary).length === 0) {
                console.log('No batch prediction data available');
                return;
            }
            
            // Create churn distribution chart
            createChurnDistributionChart(data.summary);
            
            // Create risk category chart
            createRiskCategoryChart(data.summary);
            
            // Create feature comparison chart
            if (data.summary.feature_analysis) {
                createFeatureComparisonChart(data.summary.feature_analysis);
            }
        })
        .catch(error => console.error('Error fetching batch prediction data:', error));
}

// Create a gauge chart for probability visualization
function createProbabilityGauge(probability) {
    const canvas = document.getElementById('probability-gauge');
    if (!canvas) return;
    
    // Clear existing chart
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
        existingChart.destroy();
    }
    
    // Determine risk level and color
    let color;
    if (probability < 30) {
        color = '#198754'; // success/green
    } else if (probability < 70) {
        color = '#ffc107'; // warning/yellow
    } else {
        color = '#dc3545'; // danger/red
    }
    
    // Create gauge chart
    new Chart(canvas, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [probability, 100 - probability],
                backgroundColor: [color, '#2a2a2a'],
                borderWidth: 0
            }]
        },
        options: {
            circumference: 180,
            rotation: 270,
            cutout: '70%',
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    enabled: false
                },
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Churn Probability',
                    color: '#e0e0e0',
                    font: {
                        size: 16
                    }
                },
                subtitle: {
                    display: true,
                    text: `${probability}%`,
                    color: color,
                    font: {
                        size: 24,
                        weight: 'bold'
                    },
                    padding: {
                        top: 30,
                        bottom: 0
                    }
                }
            }
        }
    });
}

// Create a radar chart for feature importance
function createFeatureImportanceChart(features) {
    const canvas = document.getElementById('feature-importance-chart');
    if (!canvas) return;
    
    // Clear existing chart
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
        existingChart.destroy();
    }
    
    // Normalize feature values for radar chart
    const normalizedFeatures = {};
    
    // Key features with their normalization ranges
    const featureRanges = {
        'CreditScore': [300, 850],
        'Age': [18, 100],
        'Tenure': [0, 10],
        'Balance': [0, 250000],
        'NumOfProducts': [1, 4],
        'HasCrCard': [0, 1],
        'IsActiveMember': [0, 1],
        'EstimatedSalary': [0, 200000]
    };
    
    Object.keys(featureRanges).forEach(feature => {
        if (features[feature] !== undefined) {
            const [min, max] = featureRanges[feature];
            const normalized = Math.min(100, Math.max(0, ((features[feature] - min) / (max - min)) * 100));
            normalizedFeatures[feature] = normalized;
        }
    });
    
    // Create radar chart
    new Chart(canvas, {
        type: 'radar',
        data: {
            labels: Object.keys(normalizedFeatures),
            datasets: [{
                label: 'Customer Profile',
                data: Object.values(normalizedFeatures),
                fill: true,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgb(75, 192, 192)',
                pointBackgroundColor: 'rgb(75, 192, 192)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(75, 192, 192)'
            }]
        },
        options: {
            scales: {
                r: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        backdropColor: 'transparent'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    angleLines: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    pointLabels: {
                        color: '#e0e0e0'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Customer Profile Analysis',
                    color: '#e0e0e0'
                },
                legend: {
                    labels: {
                        color: '#e0e0e0'
                    }
                }
            }
        }
    });
}

// Create a pie chart for churn distribution
function createChurnDistributionChart(summary) {
    const canvas = document.getElementById('churn-distribution-chart');
    if (!canvas) return;
    
    // Clear existing chart
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
        existingChart.destroy();
    }
    
    // Create pie chart
    new Chart(canvas, {
        type: 'pie',
        data: {
            labels: ['Predicted Churn', 'Predicted Stay'],
            datasets: [{
                data: [summary.predicted_churn, summary.total_customers - summary.predicted_churn],
                backgroundColor: ['#dc3545', '#198754'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Churn Distribution',
                    color: '#e0e0e0'
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#e0e0e0'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const total = summary.total_customers;
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${context.label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Create a bar chart for risk categories
function createRiskCategoryChart(summary) {
    const canvas = document.getElementById('risk-category-chart');
    if (!canvas) return;
    
    // Clear existing chart
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
        existingChart.destroy();
    }
    
    // Create bar chart
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: ['High Risk', 'Medium Risk', 'Low Risk'],
            datasets: [{
                label: 'Number of Customers',
                data: [
                    summary.high_risk_customers,
                    summary.medium_risk_customers,
                    summary.low_risk_customers
                ],
                backgroundColor: [
                    '#dc3545',
                    '#ffc107',
                    '#198754'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#e0e0e0'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#e0e0e0'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Customer Risk Categories',
                    color: '#e0e0e0'
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const total = summary.total_customers;
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${value} customers (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Create a horizontal bar chart for feature comparison
function createFeatureComparisonChart(featureAnalysis) {
    const canvas = document.getElementById('feature-comparison-chart');
    if (!canvas) return;
    
    // Clear existing chart
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
        existingChart.destroy();
    }
    
    const features = Object.keys(featureAnalysis);
    const churnedAvg = features.map(feature => featureAnalysis[feature].churned_avg);
    const notChurnedAvg = features.map(feature => featureAnalysis[feature].not_churned_avg);
    
    // Create bar chart
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [
                {
                    label: 'Churned Customers',
                    data: churnedAvg,
                    backgroundColor: 'rgba(220, 53, 69, 0.7)',
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Loyal Customers',
                    data: notChurnedAvg,
                    backgroundColor: 'rgba(25, 135, 84, 0.7)',
                    borderColor: 'rgba(25, 135, 84, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#e0e0e0'
                    }
                },
                y: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#e0e0e0'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Feature Comparison: Churned vs. Loyal Customers',
                    color: '#e0e0e0'
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#e0e0e0'
                    }
                }
            }
        }
    });
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check which page we're on by looking for specific elements
    if (document.getElementById('probability-gauge')) {
        initIndividualPredictionChart();
    }
    
    if (document.getElementById('churn-distribution-chart')) {
        initBatchPredictionCharts();
    }
});
