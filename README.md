# swe-bench-zeno

Scripts for analyzing the SWE-bench dataset with [Zeno](https://zenoml.com) and understanding model performance patterns.

## Core Analysis Scripts

### Visualization
- `acquire_data.py`: Downloads data from both swe-bench official and OpenHands repositories
- `download_issues.py`: Downloads all GitHub issues from a repository
- `visualize_results.py`: Visualization code with Zeno

### Performance Analysis
- `download_data.py`: Downloads and stores SWE-bench data locally
- `feature_engineering.py`: Extracts features including semantic embeddings
- `analyze_performance.py`: Builds and evaluates predictive models

## Performance Analysis Findings

Our analysis of OpenHands' performance on SWE-bench reveals several key patterns:

### Most Important Features
1. Problem description length (11.7%)
2. Number of imports after changes (9.1%)
3. Problem-patch semantic similarity (8.6%)
4. Patch size (7.1%)
5. Total lines changed (7.0%)
6. Variables after changes (6.6%)
7. Comments after changes (5.8%)
8. Other code structure features (44.1%)

### Key Insights

1. **Semantic Understanding**:
   - High semantic similarity doesn't guarantee success
   - Some problem clusters have up to 72.7% failure rate
   - Patch clusters show even stronger patterns (up to 85.7% failure)

2. **Technical Challenges**:
   - Struggles with longer problem descriptions (242 vs 173 words)
   - Difficulty with larger patches (136 vs 69 lines)
   - Multi-file changes are problematic (3.6 vs 2.8 files)

3. **Code Structure Patterns**:
   - Function Changes:
     * Failed cases show larger increases in parameters (+0.36 vs +0.13)
     * Failed patches have much larger function length increases (+5.96 vs +1.86)
     * Tends to fail when removing return statements (-2.0 vs +0.48)
   - Control Flow:
     * Struggles when removing control statements (-6.2 vs +1.38)
     * Better performance when preserving existing flow
   - Documentation:
     * Success correlates with adding comments (+4.57 vs +0.09)
     * Documentation changes signal understanding

4. **Complexity Patterns**:
   - Negative correlation between complexity changes and failures (-0.105)
   - Successful changes: moderate complexity increase (+26.26)
   - Failed changes: large complexity decrease (-106.32)
   - Similar maximum complexity in both cases (~400-450)

5. **Quality Signals**:
   - High variance in semantic similarity for problematic cases
   - Some failed cases show negative problem-patch similarity
   - Clear cluster-based patterns that could predict failures
   - Documentation changes as a quality indicator

For detailed analysis and recommendations, see [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md).

## Usage

### Visualization
1. Clone the repo with `acquire_data.py`
2. Add your evaluation results (must include `output.jsonl` and `report.json` files for swe-bench or `output.jsonl` for aider bench) to the `od_eval` folder
3. Add your Zeno API Key
4. Run `visualize_results.py` to generate visualizations

### Performance Analysis
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn sentence-transformers
   ```

2. Run the analysis pipeline:
   ```bash
   python download_data.py
   python feature_engineering.py
   python analyze_performance.py
   ```

## Recommendations for Improvement

1. **Type System Handling**:
   - Improve handling of complex type annotations
   - Focus on maintaining type consistency
   - Better support for generic type modifications
   - Validate type safety in generated patches

2. **Error Handling Strategy**:
   - Preserve existing error handling patterns
   - Maintain or enhance exception coverage
   - Avoid removing return statements
   - Keep error handling complexity balanced

3. **Dependency Management**:
   - Make more localized dependency changes
   - Avoid introducing circular dependencies
   - Keep import changes minimal
   - Validate dependency graph integrity

4. **Code Structure**:
   - Prefer incremental function changes
   - Maintain documentation coverage
   - Keep complexity changes moderate
   - Preserve existing control flow where possible

5. **Quality Assurance**:
   - Monitor type system modifications
   - Track error handling coverage
   - Validate dependency relationships
   - Check documentation completeness

6. **Tooling Support**:
   - Add automated type safety validation
   - Implement error coverage monitoring
   - Create dependency graph analysis tools
   - Build complexity change trackers

## Future Feature Analysis

The following feature sets have been identified as promising areas for further investigation:

### Implemented Features
1. **Basic Metrics**
   - Problem description length
   - Patch size and file modifications
   - Problem-patch semantic similarity

2. **Code Structure**
   - Function and class metrics
   - Control flow patterns
   - Variable and import analysis
   - Documentation coverage
   - Complexity scoring

### Planned Features
1. **Type System Analysis**
   - Type annotation changes
   - Type complexity (generics, unions)
   - Type inference requirements
   - Type safety impact

2. **Error Handling Coverage**
   - Exception pattern changes
   - Error case coverage
   - Exception hierarchy analysis
   - Error handling complexity

3. **Dependency Graph Analysis**
   - Function call graph changes
   - Module dependency patterns
   - Circular dependency detection
   - Import hierarchy analysis

4. **Natural Language Alignment**
   - Code identifier semantic analysis
   - Variable naming consistency
   - Function name alignment
   - Documentation-code semantic gap

5. **Context Window Requirements**
   - Required context size
   - Cross-file reference patterns
   - Scope depth analysis
   - Context switching complexity

### Other Considered Features
- API Usage Patterns
- Test Context Analysis
- Code Pattern Transitions
- Repository-Specific Patterns
- Code Style Transitions

These features were considered but deprioritized due to measurement challenges or lower expected impact on understanding performance gaps.

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- sentence-transformers
- swe-bench (local installation)
- Zeno API key (for visualization)
