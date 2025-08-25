# Test Reports in GitHub Actions

This document explains how to access test reports generated during the publish workflow.

## Overview

The publish GitHub Actions workflow now generates and preserves test reports in two ways:

1. **Downloadable Artifacts**: Complete test reports saved as artifacts
2. **Inline Results**: Test summary displayed directly in the workflow interface

## Accessing Test Reports

### Method 1: Download Artifacts

1. Navigate to the **Actions** tab in the repository
2. Click on the specific workflow run
3. Scroll down to the **Artifacts** section
4. Download the `test-reports` artifact
5. Extract the ZIP file to access:
   - HTML test reports in `build/reports/tests/`
   - XML test results in `build/test-results/`

### Method 2: View Inline Results

1. Navigate to the **Actions** tab in the repository
2. Click on the specific workflow run
3. Look for the **Test Results** section in the workflow summary
4. View test pass/fail status and detailed results directly in the interface

## Report Contents

- **HTML Reports**: Detailed test execution reports with timing, output, and failure details
- **XML Results**: JUnit-compatible test results for integration with other tools
- **Retention**: Artifacts are kept for 30 days

## Features

- Reports are generated even if tests fail (using `if: always()`)
- No impact on existing build/publish process
- Compatible with standard Gradle test reporting
- Supports both common tests and JVM-specific tests