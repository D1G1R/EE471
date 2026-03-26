

[filename] - [llm model used]

conform.py - original code
conform_gemini3.1.py - gemini 3.1 pro
confomr_haiku.py - claude haiku 4.5
conform_kimik2.5.py - moonshot ai kimi K2.5

calculator_haiku.html - claude haiku 4.5
calculator_gem3flash.html - gemini 3 flash low

# EE 471 - In-Class Exercise Week #3: Vibe Coding & LLM Performance Comparison Report

**Observations on the Original Code (`conform.py`):**
* While the original script fulfills its basic function, it uses extra lists (`intervals`) to store data and counters to keep track of directions.
* Scanning the entire list from start to finish to record intervals and then scanning those records again to print them out is an inefficient approach in terms of memory and time management.

**Model 1: Kimi K2.5 (`conform_kimik2.5.py`) Improvements:**
* **Modularity:** It significantly improved readability by breaking the code down into small, single-purpose functions (like `find_intervals`, `determine_flip_target`, `generate_commands`).
* **Type Safety (Type Hinting):** It ensured the code's reliability during the development phase by using types like `Literal` and `CapDirection`.
* **Performance Note:** Although it structurally adhered to clean code principles, the double-pass execution and unnecessary memory usage issues persisted because it didn't change the underlying core algorithm.

**Model 2: Haiku 4.5 (`conform_haiku4.5.py`) Improvements:**
* **Enterprise-Level Architecture:** It made the code much more rigid, object-oriented, and structured by utilizing `Enum` and `NamedTuple`.
* **Error Handling:** It added a robust defense mechanism that raises a `ValueError` against empty or invalid inputs (unexpected characters).
* **Comprehensive Documentation:** It produced an output highly suitable for teamwork, complete with detailed docstrings and doctest examples for each function. The main focus of this model was code maintainability rather than algorithmic speed.

**Model 3: Gemini 3.1 (`conform_gemini3.1.py`) Improvements:**
* **Algorithmic Optimization:** By grasping the core logic behind the problem (the mathematical rule that the target direction to flip is always the opposite of the first element), it reduced the code to a single-pass loop.
* **Maximum Memory Efficiency:** It achieved O(1) (constant space complexity) memory usage by completely eliminating the `intervals` list and the extra counters.
* **Simplicity:** Instead of heavy, enterprise-grade structures, it offered a high-performance, Pythonic solution stripped of unnecessary code clutter.

**Conclusion and Experience Comment:**
Vibe coding with different AI models demonstrated that the concept of "clean code" is not a one-size-fits-all truth. While the Haiku and Kimik models focused heavily on software architecture, modularity, and system safety, Gemini 3.1 directly targeted algorithmic efficiency and simplicity. It was observed that choosing the right AI assistant is critical depending on the project's scope (e.g., building a massive software architecture vs. writing a fast-running, highly optimized script).