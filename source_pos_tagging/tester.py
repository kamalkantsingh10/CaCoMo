from typing import List, Tuple, Dict, Optional
from source_pos_tagging.pos_base_corrector import POS_Tag_CaseCorrecter


class Tester:
    """
    A class to test the SentenceFormatter against a test file.
    Test file format: input</> expected_output
    """
    
    def __init__(self, test_file_path: str):
        """
        Initialize the tester with a test file.
        
        Args:
            test_file_path (str): Path to the test file
        """
        self.test_file_path = test_file_path
        self.formatter = POS_Tag_CaseCorrecter()
        self.test_cases: List[Tuple[str, str]] = []
        self._load_test_cases()
    
    def _load_test_cases(self):
        """Load test cases from the file."""
        try:
            with open(self.test_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or not '</>' in line:
                        continue
                    
                    try:
                        input_text, expected_output = line.split('</>')
                        self.test_cases.append((input_text.strip(), expected_output.strip()))
                    except ValueError:
                        print(f"Warning: Invalid format on line {line_num}: '{line}'")
        except FileNotFoundError:
            print(f"Error: Test file '{self.test_file_path}' not found.")
            raise
    
    def run_tests(self) -> Dict:
        """
        Run all test cases and return results.
        
        Returns:
            Dict: Test results including statistics and examples
        """
        results = {
            'total': len(self.test_cases),
            'passed': 0,
            'failed': 0,
            'accuracy': 0.0,
            'examples': {
                'correct': [],
                'incorrect': []
            }
        }
        
        for i, (input_text, expected_output) in enumerate(self.test_cases):
            actual_output = self.formatter.format_text(input_text)
            is_correct = actual_output == expected_output
            
            if is_correct:
                results['passed'] += 1
                if len(results['examples']['correct']) < 5:
                    results['examples']['correct'].append({
                        'input': input_text,
                        'expected': expected_output,
                        'actual': actual_output
                    })
            else:
                results['failed'] += 1
                if len(results['examples']['incorrect']) < 5:
                    results['examples']['incorrect'].append({
                        'input': input_text,
                        'expected': expected_output,
                        'actual': actual_output
                    })
        
        results['accuracy'] = (results['passed'] / results['total']) * 100 if results['total'] > 0 else 0
        return results
    
    def analyze_results(self, results: Dict) -> None:
        """
        Analyze and display test results.
        
        Args:
            results (Dict): Results from run_tests()
        """
        print("=" * 50)
        print("SENTENCE FORMATTER TEST RESULTS")
        print("=" * 50)
        print(f"Total test cases: {results['total']}")
        print(f"Passed: {results['passed']}")
        print(f"Failed: {results['failed']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print()
        
        if results['examples']['correct']:
            print("CORRECT EXAMPLES (first 5):")
            print("-" * 30)
            for example in results['examples']['correct']:
                print(f"Input:    '{example['input']}'")
                print(f"Expected: '{example['expected']}'")
                print(f"Actual:   '{example['actual']}'")
                print("✓ Correct")
                print()
        
        if results['examples']['incorrect']:
            print("INCORRECT EXAMPLES (first 5):")
            print("-" * 30)
            for example in results['examples']['incorrect']:
                print(f"Input:    '{example['input']}'")
                print(f"Expected: '{example['expected']}'")
                print(f"Actual:   '{example['actual']}'")
                print("✗ Incorrect")
                print()
    
    def analyze_errors(self, results: Dict) -> None:
        """
        Analyze common error patterns.
        
        Args:
            results (Dict): Results from run_tests()
        """
        error_types = {
            'capitalization': 0,
            'proper_nouns': 0,
            'punctuation': 0,
            'other': 0
        }
        
        # Analyze first 50 errors or all if less
        for i, (input_text, expected_output) in enumerate(self.test_cases):
            if i >= results['passed']:
                actual_output = self.formatter.format_text(input_text)
                if actual_output != expected_output:
                    # Simple error classification
                    if actual_output.lower() == expected_output.lower():
                        error_types['capitalization'] += 1
                    elif '.' in expected_output or '?' in expected_output:
                        error_types['punctuation'] += 1
                    else:
                        # Check for proper noun issues
                        expected_words = expected_output.split()
                        actual_words = actual_output.split()
                        has_proper_noun_error = False
                        for ew, aw in zip(expected_words, actual_words):
                            if ew.isalpha() and aw.isalpha():
                                if ew.istitle() != aw.istitle():
                                    has_proper_noun_error = True
                                    break
                        if has_proper_noun_error:
                            error_types['proper_nouns'] += 1
                        else:
                            error_types['other'] += 1
        
        print("\nERROR ANALYSIS:")
        print("-" * 30)
        total_errors = results['failed']
        if total_errors > 0:
            for error_type, count in error_types.items():
                percentage = (count / total_errors) * 100
                print(f"{error_type.capitalize()}: {count} ({percentage:.1f}%)")
    
    def save_error_report(self, results: Dict, output_file: str = 'error_report_pos.txt') -> None:
        """
        Save detailed error report to file.
        
        Args:
            results (Dict): Results from run_tests()
            output_file (str): Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SENTENCE FORMATTER ERROR REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total test cases: {results['total']}\n")
            f.write(f"Passed: {results['passed']}\n")
            f.write(f"Failed: {results['failed']}\n")
            f.write(f"Accuracy: {results['accuracy']:.2f}%\n\n")
            
            f.write("INCORRECT EXAMPLES:\n")
            f.write("-" * 30 + "\n")
            
            incorrect_count = 0
            for i, (input_text, expected_output) in enumerate(self.test_cases):
                actual_output = self.formatter.format_text(input_text)
                if actual_output != expected_output:
                    incorrect_count += 1
                    f.write(f"\nError #{incorrect_count}:\n")
                    f.write(f"Input:    '{input_text}'\n")
                    f.write(f"Expected: '{expected_output}'\n")
                    f.write(f"Actual:   '{actual_output}'\n")
                    
                    # Add difference analysis
                    if actual_output.lower() == expected_output.lower():
                        f.write("Type: Capitalization error\n")
                    else:
                        f.write("Type: Other error\n")

