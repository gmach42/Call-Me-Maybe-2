Encountered problems:
- Too much time and resources used to complete the decoding especially if the pc is not very performant
--> Had to implement optimizations such as caching intermediate results
- Had to create a TypeVar in the parser to handle different type of BaseModel according to the pydantic models I created. Otherwise, mypy would raise type errors.
--> Another solution to this problem would have to simply use a function for each BaseModel but the solution seems less elegant and harder to maintain.
- Messed a bit with the prompt but a simple straight to the point preprompt worked better.
- 2 caches solutions have been implemented: one for each token and one for each intermediate result of the decoding process.
- Spent quite a lot of time on the managment of special characters such as newline, tab, and escape sequences.
- The decoding process generates specific types of values (boolean, integer, string, etc.) as asked by the subject.
- Some reggex patterns have been used to match and extract specific parts of the input data.
