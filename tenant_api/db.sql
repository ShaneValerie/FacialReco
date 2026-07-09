CREATE DATABASE IF NOT EXISTS tenant_mgmt;
USE tenant_mgmt;

CREATE TABLE employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    tenant_company VARCHAR(150),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active TINYINT(1) DEFAULT 1
);

CREATE TABLE attendance_log (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT NOT NULL,
    gate VARCHAR(100) NOT NULL,
    direction VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    logged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE facial_data_reference (
    face_id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT NOT NULL,
    embedding LONGTEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE unknown_alerts (
    alert_id INT AUTO_INCREMENT PRIMARY KEY,
    gate VARCHAR(100) NOT NULL,
    snapshot_path VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
);