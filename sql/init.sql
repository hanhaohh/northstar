CREATE DATABASE IF NOT EXISTS file_activity;

USE file_activity;
-- create the table;
CREATE TABLE IF NOT EXISTS file_metadata(
    id int AUTO_INCREMENT PRIMARY KEY,
    file_name varchar(255),
    file_path varchar(255),
    file_md5 varchar(255)
);