# Security Policy

## Supported Versions

We actively support the following versions of DataBeak:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security
vulnerability in DataBeak, please report it responsibly.

### How to Report

1. **Email**: Send details to <jps@s390x.com>
1. **Subject**: Include "DataBeak Security" in the subject line
1. **Details**: Provide a detailed description of the vulnerability

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 24 hours
- **Status Update**: Within 72 hours
- **Fix Timeline**: Depends on severity (1-30 days)

### Security Best Practices

When using DataBeak:

1. **Input Validation**: Always validate CSV files before processing
1. **File Permissions**: Ensure proper file permissions for CSV files
1. **Network Security**: Use HTTPS when running in HTTP mode
1. **Access Control**: Limit MCP server access to trusted clients
1. **Regular Updates**: Keep DataBeak updated to the latest version

### Disclosure Policy

- We will acknowledge receipt of your vulnerability report
- We will provide regular updates on our progress
- We will credit you in the security advisory (unless you prefer anonymity)
- We will coordinate disclosure timing with you

### Security Features

DataBeak includes several security features:

- Input sanitization for CSV data
- File path validation to prevent directory traversal
- Memory usage limits to prevent DoS attacks
- Error handling to prevent information disclosure

Thank you for helping keep DataBeak secure!
