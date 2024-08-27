from strings_with_arrows import *
import string

#################################
# CONSTANTS
#################################

DIGITS = '0123456789'

CHARS = string.ascii_letters

#################################
# ERRORS
#################################

class Error:
    """
    A base class for handling errors.

    Attributes:
        pos_start (Position): The starting position of the error in the source code.
        pos_end (Position): The ending position of the error in the source code.
        error_name (str): The name of the error.
        details (str): Detailed description of the error.
    """

    def __init__(self, pos_start, pos_end, error_name, details):
        """
        Initializes an Error instance.

        Args:
            pos_start (Position): The starting position of the error in the source code.
            pos_end (Position): The ending position of the error in the source code.
            error_name (str): The name of the error.
            details (str): Detailed description of the error.
        """        
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details
    
    def as_string(self):
        """
        Returns a formatted string representation of the error.

        Returns:
            str: A string representation of the error including error name, details,
                 file name, line number, and a visual representation of the error in
                 the source code.
        """
        result = f'{self.error_name}: {self.details}'
        result += f'File {self.pos_start.fn}, ln {self.pos_end.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

class IllegalCharError(Error):
    """
    Exception raised for illegal characters in the source code.

    Inherits from:
        Error
    """

    def __init__(self, pos_start, pos_end, details):
        """
        Initializes an IllegalCharError instance.

        Args:
            pos_start (Position): The starting position of the error in the source code.
            pos_end (Position): The ending position of the error in the source code.
            details (str): Detailed description of the error.
        """
        super().__init__(pos_start, pos_end, 'Illegal character\n', details)

class ExpectedCharError(Error):
    """
    Exception raised when an expected character is missing in the source code.

    Inherits from:
        Error
    """

    def __init__(self, pos_start, pos_end, details):
        """
        Initializes an ExpectedCharError instance.

        Args:
            pos_start (Position): The starting position of the error in the source code.
            pos_end (Position): The ending position of the error in the source code.
            details (str): Detailed description of the error.
        """
        super().__init__(pos_start, pos_end, 'Expected character\n', details)

class InvalidSyntaxError(Error):
    """
    Exception raised for invalid syntax in the source code.

    Inherits from:
        Error
    """

    def __init__(self, pos_start, pos_end, details):
        """
        Initializes an InvalidSyntaxError instance.

        Args:
            pos_start (Position): The starting position of the error in the source code.
            pos_end (Position): The ending position of the error in the source code.
            details (str): Detailed description of the error.
        """
        super().__init__(pos_start, pos_end, 'Illegal syntax\n', details)

class RTError(Error):
    """
    Exception raised for runtime errors during the execution of the code.

    Inherits from:
        Error

    Attributes:
        context (Context): The context in which the error occurred.
    """

    def __init__(self, pos_start, pos_end, details, context):
        """
        Initializes an RTError instance.

        Args:
            pos_start (Position): The starting position of the error in the source code.
            pos_end (Position): The ending position of the error in the source code.
            details (str): Detailed description of the error.
            context (Context): The context in which the error occurred.
        """
        super().__init__(pos_start, pos_end, 'Runtime Error\n', details)
        self.context = context
    
    def as_string(self):
        """
        Returns a formatted string representation of the runtime error including
        a traceback of the call stack.

        Returns:
            str: A string representation of the runtime error including error name,
                 details, file name, line number, a visual representation of the error
                 in the source code, and a traceback of the call stack.
        """
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        """
        Generates a traceback of the call stack leading to the runtime error.

        Returns:
            str: A formatted string representing the traceback of the call stack.
        """
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
        
        return 'Traceback (most recent call last):\n' + result

#################################
# POSITION
#################################

class Position:
    """
    Represents a position within a source file.

    Attributes:
        idx (int): The current index position in the file text.
        ln (int): The current line number (0-based).
        col (int): The current column number (0-based).
        fn (str): The name of the file.
        ftxt (str): The content of the file.
    """

    def __init__(self, idx, ln, col, fn, ftxt):
        """
        Initializes a Position instance.

        Args:
            idx (int): The index position in the file text.
            ln (int): The line number (0-based).
            col (int): The column number (0-based).
            fn (str): The name of the file.
            ftxt (str): The content of the file.
        """
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        """
        Advances the position by one character.

        Args:
            current_char (str, optional): The current character at the position. If it is a newline character ('\n'),
                                          the line number is incremented and the column is reset to 0.

        Returns:
            Position: A new Position instance with updated index, line, and column values.
        """
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        """
        Creates a copy of the current Position instance.

        Returns:
            Position: A new Position instance with the same index, line, column, file name, and file text.
        """
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#################################
# TOKENS
#################################

# Reserved words in the language
SAVED_WORDS = {
    'IF', 'ELIF', 'ELSE', 'WHILE', 'ENDWHILE', 'P'
}

# Token types for integers, floats, and arithmetic operators
TT_INT      = 'INT'      # Integer literal
TT_FLOAT    = 'FLOAT'    # Floating-point literal
TT_PLUS     = 'PLUS'     # Addition operator
TT_MINUS    = 'MINUS'    # Subtraction operator
TT_MUL      = 'MUL'      # Multiplication operator
TT_DIV      = 'DIV'      # Division operator
TT_MODULO   = 'MOD'
TT_POWER    = 'POW'      # Exponentiation operator

# Token types for parentheses and end of file
TT_LPAREN   = 'LPAREN'   # Left parenthesis
TT_RPAREN   = 'RPAREN'   # Right parenthesis
TT_EOF      = 'EOF'      # End of file token

# Token types for min and max operations
TT_MIN      = 'MIN'      # Minimum function
TT_MAX      = 'MAX'      # Maximum function

# Token types for punctuation and assignment
TT_COMMA    = 'COMMA'    # Comma
TT_ASSIGN   = 'ASSIGN'   # Assignment operator
TT_EQUAL    = 'EQUAL'    # Equality operator
TT_NEQUAL   = 'NEQUAL'   # Not equal operator

# Token types for logical operators
TT_AND      = 'AND'      # Logical AND
TT_OR       = 'OR'       # Logical OR
TT_NOT      = 'NOT'      # Logical NOT

# Token types for comparison operators
TT_GT       = 'GTHEN'    # Greater than operator
TT_LT       = 'LTHEN'    # Less than operator
TT_GTE      = 'GTEQUAL'  # Greater than or equal to operator
TT_LTE      = 'LTEQUAL'  # Less than or equal to operator

# Token types for variables, keywords, and string literals
TT_VAR      = 'VAR'      # Variable identifier
TT_KEYWORD  = 'KEYWORD'  # Reserved keyword
TT_SNGLQTE  = 'SNGLQUOTE' # Single quote for strings
TT_DBLQTE   = 'DBLQUOTE'  # Double quote for strings
TT_NEWLINE  = 'NEWLINE'   # Newline character
TT_STRING   = 'STR'      # String literal

# Token types for control flow and special characters
TT_IF       = 'IF'       # 'IF' keyword
TT_ELIF     = 'ELIF'     # 'ELIF' keyword
TT_ELSE     = 'ELSE'     # 'ELSE' keyword
TT_WHILE    = 'WHILE'    # 'WHILE' keyword
TT_COLON    = 'COLON'    # Colon character

class Token:
    """
    Represents a token in the source code.

    Attributes:
        type (str): The type of the token (e.g., 'INT', 'PLUS').
        value (any): The value of the token, if applicable (e.g., 42, '+').
        pos_start (Position): The starting position of the token in the source code.
        pos_end (Position): The ending position of the token in the source code.
    """

    def __init__(self, type_, value=None, pos_start=0, pos_end=0):
        """
        Initializes a Token instance.

        Args:
            type_ (str): The type of the token.
            value (any, optional): The value of the token, if applicable.
            pos_start (Position or int, optional): The starting position of the token.
            pos_end (Position or int, optional): The ending position of the token.
        """
        self.type = type_
        self.value = value

        # Initialize pos_start and pos_end
        if pos_start:
            self.pos_start = pos_start.copy()  # Copy the starting position
            self.pos_end = pos_start.copy()    # Copy the ending position from the start
            self.pos_end.advance()             # Advance the ending position by one

        if pos_end:
            self.pos_end = pos_end             # Set the ending position if provided

    def matches(self, type, value):
        """
        Checks if the token matches the given type and value.

        Args:
            type (str): The type to check against.
            value (any): The value to check against.

        Returns:
            bool: True if the token's type and value match the given type and value, False otherwise.
        """
        return self.type == type and self.value == value

    def __repr__(self):
        """
        Returns a string representation of the token.

        Returns:
            str: A string showing the token's type and value (if available).
        """
        if self.value:
            return f'{self.type}:{self.value}'  # Return type and value if value exists
        return f'{self.type}'  # Return only type if value is not set
    
#################################
# LEXER
#################################
    
class Lexer:
    """
    A lexer for tokenizing input text.

    Attributes:
        fn (str): The name of the file from which the text is read.
        text (str): The input text to be tokenized.
        pos (Position): The current position in the text.
        current_char (str): The current character being examined.
    """

    def __init__(self, fn, text):
        """
        Initializes the Lexer with the given file name and text.

        Args:
            fn (str): The name of the file.
            text (str): The input text to be tokenized.
        """
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)  # Initialize the position
        self.current_char = None
        self.advance()  # Move to the first character

    def advance(self, steps=1):
        """
        Advances the current position by the specified number of steps.

        Args:
            steps (int): The number of steps to advance (default is 1).
        """
        for _ in range(steps):
            self.pos.advance(self.current_char)
            self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def peek(self, steps=1):
        """
        Peeks ahead in the text by the specified number of steps.

        Args:
            steps (int): The number of steps to peek ahead (default is 1).

        Returns:
            str or None: The character at the peek position or None if out of bounds.
        """
        peek_pos = self.pos.idx + steps
        if peek_pos >= len(self.text):
            return None
        return self.text[peek_pos]

    def make_tokens(self):
        """
        Tokenizes the input text and returns a list of tokens and an optional error.

        Returns:
            tuple: A tuple containing a list of tokens and an error (if any).
        """
        tokens = []

        while self.current_char is not None:

            if self.current_char in ' \t':
                self.advance()  # Skip whitespace

            elif self.current_char in DIGITS:
                tokens.append(self.make_number())  # Handle numbers

            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()

            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()

            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()

            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()

            elif self.current_char == '%':
                tokens.append(Token(TT_MODULO, pos_start=self.pos))
                self.advance()

            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()

            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()

            elif self.current_char == '^':
                tokens.append(Token(TT_POWER, pos_start=self.pos))
                self.advance()

            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()

            elif self.current_char == ':':
                tokens.append(Token(TT_COLON, pos_start=self.pos))
                self.advance()

            elif self.current_char == 'M' and self.peek() == 'i':
                if self.peek(2) == 'n' and self.peek(3) == "(":
                    tokens.append(Token(TT_MIN, pos_start=self.pos))
                    self.advance(3)  # Skip 'Min('
                else:
                    return [], ExpectedCharError(self.pos, self.pos, "Expected '('")

            elif self.current_char == 'M' and self.peek() == 'a':
                if self.peek(2) == 'x' and self.peek(3) == "(":
                    tokens.append(Token(TT_MAX, pos_start=self.pos))
                    self.advance(3)  # Skip 'Max('
                else:
                    return [], ExpectedCharError(self.pos, self.pos, "Expected '('")

            elif self.current_char in CHARS:
                tokens.append(self.make_variable())  # Handle variables

            elif self.current_char == '=':
                if self.peek() == '=':
                    tokens.append(Token(TT_EQUAL, pos_start=self.pos))
                    self.advance(2)
                else:
                    tokens.append(Token(TT_ASSIGN, pos_start=self.pos))
                    self.advance()

            elif self.current_char == '!':
                if self.peek() == '=':
                    tokens.append(Token(TT_NEQUAL, pos_start=self.pos))
                    self.advance(2)
                else:
                    tokens.append(Token(TT_NOT, pos_start=self.pos))
                    self.advance()

            elif self.current_char == '>':
                if self.peek() == '=':
                    tokens.append(Token(TT_GTE, pos_start=self.pos))
                    self.advance(2)
                else:
                    tokens.append(Token(TT_GT, pos_start=self.pos))
                    self.advance()

            elif self.current_char == '<':
                if self.peek() == '=':
                    tokens.append(Token(TT_LTE, pos_start=self.pos))
                    self.advance(2)
                else:
                    tokens.append(Token(TT_LT, pos_start=self.pos))
                    self.advance()

            elif self.current_char == '&':
                tokens.append(Token(TT_AND, pos_start=self.pos))
                self.advance()

            elif self.current_char == '|' and self.peek() == '|':
                tokens.append(Token(TT_OR, pos_start=self.pos))
                self.advance(2)

            elif self.current_char in "\"'":
                result, error = self.make_string()  # Handle string literals
                if error:
                    return [], error  # Return error if there is one
                tokens.append(result)

            elif self.current_char == '\n':
                tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
                self.advance()

            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, f"Illegal character '{char}'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        """
        Creates a number token from the current text.

        Returns:
            Token: A token representing an integer or floating-point number.
        """
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()
        
        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def make_variable(self):
        """
        Creates a variable token from the current text.

        Returns:
            Token: A token representing a variable or keyword.
        """
        var_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            var_str += self.current_char
            self.advance()

        # Check if the variable name is a keyword
        if var_str.upper() in SAVED_WORDS:
            return Token(TT_KEYWORD, var_str, pos_start, self.pos)
        else:
            return Token(TT_VAR, var_str, pos_start, self.pos)

    def make_string(self):
        """
        Creates a string token from the current text.

        Returns:
            tuple: A tuple containing a token representing the string and an optional error.
        """
        str_val = ''
        pos_start = self.pos.copy()
        quote_type = self.current_char  # Save whether it's a single or double quote

        self.advance()  # Skip the opening quote
        while self.current_char is not None and self.current_char != quote_type:
            str_val += self.current_char
            self.advance()

        # If we reached the end of the input without finding the closing quote
        if self.current_char != quote_type:
            return None, ExpectedCharError(pos_start, self.pos, f"Expected closing {quote_type} for string")

        self.advance()  # Skip the closing quote
        return Token(TT_STRING, str_val, pos_start, self.pos), None
        
#################################
# NODES
#################################
        
class NumberNode:
    """
    A node representing a number.

    Attributes:
        tok (Token): The token representing the number.
        pos_start (Position): The starting position of the token.
        pos_end (Position): The ending position of the token.
    """
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    
    def __repr__(self):
        return f'{self.tok}'

class BinOpNode:
    """
    A node representing a binary operation.

    Attributes:
        left_node (Node): The left operand of the operation.
        op_tok (Token): The token representing the operator.
        right_node (Node): The right operand of the operation.
        pos_start (Position): The starting position of the operation.
        pos_end (Position): The ending position of the operation.
    """
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
    """
    A node representing a unary operation.

    Attributes:
        op_tok (Token): The token representing the operator.
        node (Node): The operand of the operation.
        pos_start (Position): The starting position of the operation.
        pos_end (Position): The ending position of the operation.
    """
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'

class MinNode:
    """
    A node representing the Min function.

    Attributes:
        left_node (Node): The first argument to the Min function.
        right_node (Node): The second argument to the Min function.
        pos_start (Position): The starting position of the function.
        pos_end (Position): The ending position of the function.
    """
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'Min({self.left_node}, {self.right_node})'

class MaxNode:
    """
    A node representing the Max function.

    Attributes:
        left_node (Node): The first argument to the Max function.
        right_node (Node): The second argument to the Max function.
        pos_start (Position): The starting position of the function.
        pos_end (Position): The ending position of the function.
    """
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'Max({self.left_node}, {self.right_node})'

class LogicalOpNode:
    """
    A node representing a logical operation.

    Attributes:
        left_node (Node): The left operand of the logical operation.
        op_tok (Token): The token representing the logical operator.
        right_node (Node): The right operand of the logical operation.
        pos_start (Position): The starting position of the operation.
        pos_end (Position): The ending position of the operation.
    """
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node} {self.op_tok} {self.right_node})'

class UnaryLogicalOpNode:
    """
    A node representing a unary logical operation.

    Attributes:
        op_tok (Token): The token representing the logical operator.
        node (Node): The operand of the logical operation.
        pos_start (Position): The starting position of the operation.
        pos_end (Position): The ending position of the operation.
    """
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.node.pos_start
        self.pos_end = self.node.pos_end

    def __repr__(self):
        return f'({self.op_tok} {self.node})'

class CompOpNode:
    """
    A node representing a comparison operation.

    Attributes:
        left_node (Node): The left operand of the comparison.
        op_tok (Token): The token representing the comparison operator.
        right_node (Node): The right operand of the comparison.
        pos_start (Position): The starting position of the comparison.
        pos_end (Position): The ending position of the comparison.
    """
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node} {self.op_tok} {self.right_node})'

class StatementsNode:
    """
    A node representing a list of statements.

    Attributes:
        statements (list): A list of statement nodes.
        pos_start (Position): The starting position of the first statement.
        pos_end (Position): The ending position of the last statement.
    """
    def __init__(self, statements):
        self.statements = [stmt for stmt in statements if not isinstance(stmt, EmptyNode)]
        self.pos_start = self.statements[0].pos_start if self.statements else None
        self.pos_end = self.statements[-1].pos_end if self.statements else None

    def __repr__(self):
        return f"{self.statements}"

class VariableNode:
    """
    A node representing a variable assignment.

    Attributes:
        var_name (Token): The token representing the variable name.
        value (Node): The value assigned to the variable.
        pos_start (Position): The starting position of the assignment.
        pos_end (Position): The ending position of the assignment.
    """
    def __init__(self, var_name, value):
        self.var_name = var_name
        self.value = value

        self.pos_start = self.var_name.pos_start
        self.pos_end = self.var_name.pos_end if value is None else self.value.pos_end
    
    def __repr__(self):
        return f'({self.var_name} = {self.value})'

class StringNode:
    """
    A node representing a string.

    Attributes:
        tok (Token): The token representing the string.
        pos_start (Position): The starting position of the string.
        pos_end (Position): The ending position of the string.
    """
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    
    def __repr__(self):
        return f'{self.tok}'

class IfNode:
    """
    A node representing an if statement.

    Attributes:
        cases (list): A list of (condition, body) tuples for the if and elif cases.
        else_case (Node): The else case body node.
        pos_start (Position): The starting position of the if statement.
        pos_end (Position): The ending position of the if statement.
    """
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start if self.cases and self.cases[0][0] else None
        self.pos_end = (self.else_case.pos_end if self.else_case else self.cases[-1][-1].pos_end) if self.cases else None

    def __repr__(self):
        return f'IfNode({self.cases}, {self.else_case})'

class WhileNode:
    """
    A node representing a while loop.

    Attributes:
        condition (Node): The condition of the while loop.
        body (Node): The body of the while loop.
        pos_start (Position): The starting position of the while loop.
        pos_end (Position): The ending position of the while loop.
    """
    def __init__(self, condition, body, pos_start, pos_end):
        self.condition = condition
        self.body = body
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f'(WHILE {self.condition} DO {self.body})'

class PrintNode:
    """
    A node representing a print statement.

    Attributes:
        node_to_print (Node): The node representing the expression to be printed.
        pos_start (Position): The starting position of the print statement.
        pos_end (Position): The ending position of the print statement.
    """
    def __init__(self, node_to_print, pos_start, pos_end):
        self.node_to_print = node_to_print
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f'(PRINT {self.node_to_print})'

class EmptyNode:
    """
    A placeholder node representing an empty statement or value.

    Attributes:
        pos_start (Position): The starting position of the empty node.
        pos_end (Position): The ending position of the empty node.
    """
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return 'EmptyNode()'

#################################
# PARSER RESULT
#################################
    
class ParseResult:
    """
    Represents the result of parsing an input, encapsulating either a successful parse result or an error.

    Attributes:
        error (Error): The error encountered during parsing, if any.
        node (Node): The node resulting from a successful parse, if any.

    Methods:
        register(res):
            Registers the result of a parsing operation and propagates any errors.
        success(node):
            Sets the result to a successful parse with the given node and returns the instance.
        failure(error):
            Sets the result to a failed parse with the given error and returns the instance.
    """
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        """
        Registers the result of a parsing operation.

        Args:
            res (ParseResult): The result of a parsing operation, which may contain an error or a node.

        Returns:
            Node: The node from the result if parsing was successful; otherwise, propagates the error.
        """
        if isinstance(res, ParseResult):
            if res.error:
                self.error = res.error
            return res.node
        return res

    def success(self, node):
        """
        Sets the result to a successful parse with the given node.

        Args:
            node (Node): The node resulting from a successful parse.

        Returns:
            ParseResult: The current instance with the success state set.
        """
        self.node = node
        return self

    def failure(self, error):
        """
        Sets the result to a failed parse with the given error.

        Args:
            error (Error): The error encountered during parsing.

        Returns:
            ParseResult: The current instance with the failure state set.
        """
        self.error = error
        return self

#################################
# PARSER
#################################

class Parser:
    """
    A class that parses tokens into an abstract syntax tree (AST) for a custom programming language.

    This parser handles various constructs including:
    - Factors: literals (integers, floats, strings), unary operations, function calls, and variables.
    - Expressions: logical operations (AND, OR), comparison operations (==, !=, >, <, >=, <=), and arithmetic operations (+, -, *, /, ^).
    - Statements: variable assignments and function calls.
    - Parentheses for grouping expressions.
    
    The parser uses a recursive descent approach to process tokens and build an AST based on the syntax rules of the language.

    Attributes:
        tokens (list): The list of tokens to be parsed.
        current_tok (Token): The current token being processed.
    
    Methods:
        factor: Parses a factor, including literals, unary operations, function calls, and variables.
        comp_expr: Parses a comparison expression involving comparison operators.
        expr: Parses an expression involving logical operations.
        term: Parses a term involving arithmetic operations.
        logic_expr: Parses a logical expression involving logical operators.
        arith_expr: Parses an arithmetic expression involving arithmetic operators.
        bin_op: Parses binary operations involving specified operators.
        logical_op: Parses logical operations involving specified logical operators.
        comp_op: Parses comparison operations involving specified comparison operators.
    """

    def __init__(self, tokens):
        """
        Initializes the Parser with a list of tokens.

        Parameters:
            tokens (list): A list of tokens to parse.
        """
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        """
        Advances to the next token in the token list.

        Returns:
            Token: The current token after advancing.
        """
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok
    
    def peek(self, steps=1):
        """
        Peeks at the token a certain number of steps ahead from the current token.

        Parameters:
            steps (int): The number of steps ahead to look.

        Returns:
            Token or None: The token at the peek position, or None if out of range.
        """
        peek_idx = self.tok_idx + steps
        if peek_idx >= len(self.tokens):
            return None
        return self.tokens[peek_idx]

    def parse(self):
        """
        Parses a single statement, which could be a print statement, a loop, a condition, an assignment, or an expression.

        Returns:
            ParseResult: The result of parsing the statement, including the constructed node or an error.
        """
        res = self.statements()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '+', '-', '*', '/' or comparison operator"
            ))
        return res

    def statements(self):
        """
        Parses a single statement, which could be a print statement, a loop, a condition, an assignment, or an expression.

        Returns:
            ParseResult: The result of parsing the statement, including the constructed node or an error.
        """
        res = ParseResult()
        statements = []

        while self.current_tok.type != TT_EOF:
            if self.current_tok.type == TT_NEWLINE:
                res.register(self.advance())
            else:
                statement = res.register(self.statement())
                if res.error:
                    return res
                statements.append(statement)

        return res.success(StatementsNode(statements))

    def statement(self):
        """
        Parses a single statement, which could be a print statement, a loop, a condition, an assignment, or an expression.

        Returns:
            ParseResult: The result of parsing the statement, including the constructed node or an error.
        """
        res = ParseResult()

        if self.current_tok.type == TT_NEWLINE:
            pos_start = self.current_tok.pos_start.copy()
            pos_end = self.current_tok.pos_end.copy()
            res.register(self.advance())
            return res.success(EmptyNode(pos_start, pos_end))

        elif self.current_tok.matches(TT_KEYWORD, 'P'):
            pos_start = self.current_tok.pos_start
            res.register(self.advance())
            
            if self.current_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '('"
                ))
            
            res.register(self.advance())
            
            expr = res.register(self.expr())
            if res.error:
                return res
            
            if self.current_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')'"
                ))
            
            pos_end = self.current_tok.pos_end
            res.register(self.advance())
            
            return res.success(PrintNode(expr, pos_start, pos_end))

        elif self.current_tok.matches(TT_KEYWORD, 'WHILE'):
            while_loop = res.register(self.while_expr())
            if res.error:
                return res
            return res.success(while_loop)

        elif self.current_tok.matches(TT_KEYWORD, 'IF'):
            if_stmt = res.register(self.if_expr())
            if res.error:
                return res
            return res.success(if_stmt)
        
        elif self.current_tok.matches(TT_KEYWORD, 'ELIF'):
            elif_stmt = res.register(self.if_expr())  
            if res.error:
                return res
            return res.success(elif_stmt)
        
        elif self.current_tok.matches(TT_KEYWORD, 'ELSE'):
            else_stmt = res.register(self.if_expr())  
            if res.error:
                return res
            return res.success(else_stmt)

        elif self.current_tok.type == TT_VAR and self.peek().type == TT_ASSIGN:
            # This is an assignment
            return self.expr()

        elif self.current_tok.type in (TT_INT, TT_FLOAT, TT_STRING):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected variable assignment, not standalone value"
            ))

        return self.expr()
    
    def while_expr(self):
        """
        Parses a while loop statement, including its condition and body.

        Returns:
            ParseResult: The result of parsing the while loop, including the constructed WhileNode.
        """
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        res.register(self.advance()) 
        condition = res.register(self.expr())
        if res.error: 
            return res
        
        if not self.current_tok.type == TT_COLON:
            print(f"Error: Expected ':', but got {self.current_tok}")
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected ':'"
            ))
        
        res.register(self.advance())
        if self.current_tok.type != TT_NEWLINE:
            print(f"Error: Expected 'NEWLINE', but got {self.current_tok}")
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'NEWLINE'"
            ))

        res.register(self.advance())

        loop_body = []
        while not self.current_tok.matches(TT_KEYWORD, 'ENDWHILE'):
            stmt = res.register(self.statement())
            if res.error: 
                    return res
            loop_body.append(stmt)

            if self.current_tok.type != TT_NEWLINE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'NEWLINE'"
                ))

            res.register(self.advance())
        
        if not self.current_tok.matches(TT_KEYWORD, 'ENDWHILE'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'ENDWHILE' "
            ))
        res.register(self.advance())
        pos_end = self.current_tok.pos_end.copy()
        return res.success(WhileNode(condition, loop_body, pos_start, pos_end))
    
    def if_expr(self):
        """
        Parses an if-elif-else statement, including its conditions and associated bodies.

        Returns:
            ParseResult: The result of parsing the if expression, including the constructed IfNode.
        """
        res = ParseResult()
        cases = []
        else_case = None

        if self.current_tok.matches(TT_KEYWORD, 'IF'):
            res.register(self.advance())

            condition = res.register(self.expr())
            if res.error: 
                return res
            
            if not self.current_tok.type == TT_COLON:
                print(f"Error: Expected ':', but got {self.current_tok}")
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ':'"
                ))

            res.register(self.advance())
            if self.current_tok.type != TT_NEWLINE:
                print(f"Error: Expected 'NEWLINE', but got {self.current_tok}")
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'NEWLINE'"
                ))

            res.register(self.advance())

            body = res.register(self.statement())  
            if res.error: 
                return res
            cases.append((condition, body)) 
            res.register(self.advance())          

        while self.current_tok.matches(TT_KEYWORD, 'ELIF'):
            res.register(self.advance())

            condition = res.register(self.expr())
            if res.error: 
                return res

            if not self.current_tok.type == TT_COLON:
                print(f"Error: Expected ':', but got {self.current_tok}")
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ':'"
                ))

            res.register(self.advance())

            if self.current_tok.type != TT_NEWLINE:
                print(f"Error: Expected 'NEWLINE', but got {self.current_tok}")
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'NEWLINE'"
                ))

            res.register(self.advance())
            body = res.register(self.statement())
            if res.error: 
                return res
            cases.append((condition, body))
            res.register(self.advance())

        if self.current_tok.matches(TT_KEYWORD, 'ELSE'):
            res.register(self.advance())

            if not self.current_tok.type == TT_COLON:
                print(f"Error: Expected ':', but got {self.current_tok}")
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ':' "
                ))

            res.register(self.advance())

            if self.current_tok.type != TT_NEWLINE:
                print(f"Error: Expected 'NEWLINE', but got {self.current_tok}")
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'NEWLINE' "
                ))

            res.register(self.advance())

            else_case = res.register(self.statement())       
            if res.error: 
                return res  
            cases.append((None, else_case))           

        return res.success(IfNode(cases, else_case))

    def factor(self):
        """
        Parses a factor, which can be a number, a string, a unary operation, or an expression within parentheses.
        It also handles function calls for 'Min' and 'Max' and variable handling for both assignments and usages.

        Returns:
            ParseResult: The result of parsing the factor, including the constructed node or an error.
        """
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor()) 
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor))
        
        elif tok.type == TT_NOT:
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryLogicalOpNode(tok, factor))

        elif tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))
        
        elif tok.type in (TT_STRING):
            res.register(self.advance())
            return res.success(StringNode(tok))
        
        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr) 
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')' "
                ))

        elif tok.type in (TT_MIN, TT_MAX):
            tok_str = 'Min' if tok.type == TT_MIN else 'Max'
            res.register(self.advance())
            if self.current_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '(' after '{tok_str}'"
                ))
            
            res.register(self.advance())  # Skip the '(' char
            left_expr = res.register(self.expr())
            if res.error:
                return res
            
            if self.current_tok.type != TT_COMMA:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ',' after first argument in '{tok_str}'"
                ))
            
            res.register(self.advance())  # Skip the ',' char
            right_expr = res.register(self.expr())
            if res.error:
                return res
            
            if self.current_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ')' after second argument in '{tok_str}'"
                ))

            res.register(self.advance())
            return res.success(
                MinNode(left_expr, right_expr)) if tok.type == TT_MIN else res.success(
                    MaxNode(left_expr, right_expr))
        
        elif tok.type in (TT_VAR):
            var_name = tok
            res.register(self.advance())

            # Check if it's a variable assignment
            if self.current_tok.type == TT_ASSIGN:
                res.register(self.advance())
                expr_value = res.register(self.expr())  # Parse the value as an expression
                if res.error:
                    return res
                return res.success(VariableNode(var_name, expr_value))
            # Otherwise, it's a variable usage
            else:
                return res.success(VariableNode(var_name, None))

        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            "Expected int or float"
        ))

    def comp_expr(self):
        """
        Parses a comparison expression involving comparison operators (==, !=, >, <, >=, <=).

        Returns:
            ParseResult: The result of parsing the comparison expression, including the constructed node or an error.
        """
        return self.comp_op(self.arith_expr, (TT_EQUAL, TT_NEQUAL, TT_GT, TT_LT, TT_GTE, TT_LTE))

    def expr(self):
        """
        Parses an expression involving logical operations (AND, OR).

        Returns:
            ParseResult: The result of parsing the expression, including the constructed node or an error.
        """
        return self.bin_op(self.logic_expr, (TT_OR, TT_AND))

    def term(self):
        """
        Parses a term involving arithmetic operations (*, /, ^).

        Returns:
            ParseResult: The result of parsing the term, including the constructed node or an error.
        """
        return self.bin_op(self.factor, (TT_MUL, TT_DIV, TT_MODULO, TT_POWER))
    
    def logic_expr(self):
        """
        Parses a logical expression involving logical operations (AND, OR).

        Returns:
            ParseResult: The result of parsing the logical expression, including the constructed node or an error.
        """
        return self.logical_op(self.comp_expr, (TT_AND, TT_OR))
    
    def arith_expr(self):
        """
        Parses an arithmetic expression involving arithmetic operations (+, -).

        Returns:
            ParseResult: The result of parsing the arithmetic expression, including the constructed node or an error.
        """
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def bin_op(self, func, ops):
        """
        Parses binary operations involving specified operators.

        Parameters:
            func (function): A function that returns the result of parsing the left operand.
            ops (tuple): A tuple of token types representing the operators.

        Returns:
            ParseResult: The result of parsing the binary operation, including the constructed node or an error.
        """
        res = ParseResult()
        left = res.register(func())
        if res.error:
            return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error:
                return res
            left = BinOpNode(left, op_tok, right)
        
        return res.success(left)     

    def logical_op(self, func, ops):
        """
        Parses logical operations involving specified logical operators.

        Parameters:
            func (function): A function that returns the result of parsing the left operand.
            ops (tuple): A tuple of token types representing the logical operators.

        Returns:
            ParseResult: The result of parsing the logical operation, including the constructed node or an error.
        """
        res = ParseResult()
        left = res.register(func())
        if res.error:
            return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error:
                return res
            left = LogicalOpNode(left, op_tok, right)
        
        return res.success(left)
    
    def comp_op(self, func, ops):
        """
        Parses comparison operations involving specified comparison operators.

        Parameters:
            func (function): A function that returns the result of parsing the left operand.
            ops (tuple): A tuple of token types representing the comparison operators.

        Returns:
            ParseResult: The result of parsing the comparison operation, including the constructed node or an error.
        """
        res = ParseResult()
        left = res.register(func())
        if res.error:
            return res

        if self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error:
                return res
            left = CompOpNode(left, op_tok, right)
        
        return res.success(left)

#################################
# RUNTIME RESULT
#################################
    
class RTResult:
    """
    A class to handle the result of an operation in the runtime environment, encapsulating the value and error state.

    This class is used to manage the outcomes of operations performed by the interpreter. It supports registering results,
    determining if an error has occurred, and handling successful or failed results.

    Attributes:
        value (any): The value of the result, or None if there was an error.
        error (Exception or None): The error encountered during the operation, or None if there was no error.

    Methods:
        register(res): Registers a result from another operation. If there is an error, it sets the error attribute.
        success(value): Sets the result as successful with the given value and returns the instance.
        failure(error):Sets the result as a failure with the given error and returns the instance.
        should_return(): Determines if the result indicates an error. Returns True if there is an error, otherwise False.
    """

    def __init__(self):
        """
        Initializes a new `RTResult` instance with `value` and `error` set to `None`.
        """
        self.value = None
        self.error = None

    def register(self, res):
        """
        Registers the result of another operation.

        If the `res` object has an error, this method updates the `error` attribute of the current `RTResult` instance.

        Args:
            res (RTResult): The result object to register.

        Returns:
            any: The value from `res`, or None if there was an error.
        """
        if res.error:
            self.error = res.error
        return res.value
    
    def success(self, value):
        """
        Sets the result as successful with the given `value`.

        This method updates the `value` attribute and returns the current `RTResult` instance.

        Args:
            value (any): The value to set as the successful result.

        Returns:
            RTResult: The current `RTResult` instance with the `value` set.
        """
        self.value = value
        return self
    
    def failure(self, error):
        """
        Sets the result as a failure with the given `error`.

        This method updates the `error` attribute and returns the current `RTResult` instance.

        Args:
            error (Exception): The error to set as the failure result.

        Returns:
            RTResult: The current `RTResult` instance with the `error` set.
        """
        self.error = error
        return self
    
    def should_return(self):
        """
        Checks if the result indicates an error.

        Returns:
            bool: `True` if the `error` attribute is not None, otherwise `False`.
        """
        return self.error is not None
    
#################################
# VALUES
#################################

class Number:
    """
    A class representing a numeric value in the interpreter.

    This class encapsulates a numeric value and provides methods for performing arithmetic operations,
    setting positional and contextual information, and copying the instance.

    Attributes:
        value (float or int): The numeric value.
        pos_start (Position or None): The start position of the token in the source code.
        pos_end (Position or None): The end position of the token in the source code.
        context (Context or None): The context in which this number is used.

    Methods:
        set_pos(pos_start=None, pos_end=None):
            Sets the positional information of the number.

        set_context(context=None):
            Sets the context of the number.

        copy():
            Creates a copy of the current `Number` instance with the same value, positional, and contextual information.

        added_to(other):
            Adds the current number to another `Number` instance and returns the result as a new `Number` instance.
            Returns a tuple (result, error).

        subbed_by(other):
            Subtracts another `Number` instance from the current number and returns the result as a new `Number` instance.
            Returns a tuple (result, error).

        powed_by(other):
            Raises the current number to the power of another `Number` instance and returns the result as a new `Number` instance.
            Returns a tuple (result, error).

        multed_by(other):
            Multiplies the current number by another `Number` instance and returns the result as a new `Number` instance.
            Returns a tuple (result, error).

        dived_by(other):
            Divides the current number by another `Number` instance and returns the result as a new `Number` instance.
            Returns a tuple (result, error). If division by zero is attempted, returns None and an `RTError`.

        __str__():
            Returns a string representation of the number.
    """
    
    def __init__(self, value):
        """
        Initializes a new `Number` instance with the given `value`.

        Args:
            value (float or int): The numeric value.
        """
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        """
        Sets the positional information for the number.

        Args:
            pos_start (Position or None): The start position of the token.
            pos_end (Position or None): The end position of the token.

        Returns:
            Number: The current `Number` instance with updated positional information.
        """
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        """
        Sets the context for the number.

        Args:
            context (Context or None): The context in which this number is used.

        Returns:
            Number: The current `Number` instance with updated context.
        """
        self.context = context
        return self

    def copy(self):
        """
        Creates a copy of the current `Number` instance.

        Returns:
            Number: A new `Number` instance with the same value, positional, and contextual information as the original.
        """
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def added_to(self, other):
        """
        Adds the current number to another `Number` instance.

        Args:
            other (Number): The `Number` instance to add.

        Returns:
            tuple: A tuple (result, error). The result is a new `Number` instance with the sum, or None if there's an error.
        """
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        
    def subbed_by(self, other):
        """
        Subtracts another `Number` instance from the current number.

        Args:
            other (Number): The `Number` instance to subtract.

        Returns:
            tuple: A tuple (result, error). The result is a new `Number` instance with the difference, or None if there's an error.
        """
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
    
    def powed_by(self, other):
        """
        Raises the current number to the power of another `Number` instance.

        Args:
            other (Number): The `Number` instance representing the exponent.

        Returns:
            tuple: A tuple (result, error). The result is a new `Number` instance with the power result, or None if there's an error.
        """
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        
    def multed_by(self, other):
        """
        Multiplies the current number by another `Number` instance.

        Args:
            other (Number): The `Number` instance to multiply by.

        Returns:
            tuple: A tuple (result, error). The result is a new `Number` instance with the product, or None if there's an error.
        """
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        
    def dived_by(self, other):
        """
        Divides the current number by another `Number` instance.

        Args:
            other (Number): The `Number` instance to divide by.

        Returns:
            tuple: A tuple (result, error). The result is a new `Number` instance with the quotient, or None and an `RTError` if there's an error.
        """
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    "Division by zero is undefined\n",
                    self.context
                )
            
            return Number(self.value / other.value).set_context(self.context), None
    
    def modulo(self, other):
        """
        Performs the modulo operation of the current number with another `Number` instance.

        Args:
            other (Number): The `Number` instance to perform the modulo operation with.

        Returns:
            tuple: A tuple (result, error). The result is a new `Number` instance with the remainder, or None and an `RTError` if there's an error.
        """
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    "Modulo by zero is undefined\n",
                    self.context
                )
            
            return Number(self.value % other.value).set_context(self.context), None
        
    def __str__(self):
        """
        Returns a string representation of the number.

        Returns:
            str: A string representation of the `value`. If the value is a boolean, it returns its string form; otherwise, it returns the numeric string.
        """
        if isinstance(self.value, bool):
            return str(self.value)
        return str(self.value)

class Strings:
    """
    A class representing a string value in the interpreter.

    This class encapsulates a string value and provides methods for string concatenation,
    setting positional and contextual information, and copying the instance.

    Attributes:
        value (str): The string value.
        pos_start (Position or None): The start position of the token in the source code.
        pos_end (Position or None): The end position of the token in the source code.
        context (Context or None): The context in which this string is used.

    Methods:
        set_pos(pos_start=None, pos_end=None):
            Sets the positional information of the string.

        set_context(context=None):
            Sets the context of the string.

        copy():
            Creates a copy of the current `Strings` instance with the same value, positional, and contextual information.

        concat(other):
            Concatenates the current string with another `Strings` instance and returns the result as a new `Strings` instance.
            Returns the concatenated `Strings` instance, or None if the other instance is not of type `Strings`.

        __repr__():
            Returns a string representation of the string value.
    """
    
    def __init__(self, value):
        """
        Initializes a new `Strings` instance with the given `value`.

        Args:
            value (str): The string value.
        """
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        """
        Sets the positional information for the string.

        Args:
            pos_start (Position or None): The start position of the token.
            pos_end (Position or None): The end position of the token.

        Returns:
            Strings: The current `Strings` instance with updated positional information.
        """
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        """
        Sets the context for the string.

        Args:
            context (Context or None): The context in which this string is used.

        Returns:
            Strings: The current `Strings` instance with updated context.
        """
        self.context = context
        return self
    
    def copy(self):
        """
        Creates a copy of the current `Strings` instance.

        Returns:
            Strings: A new `Strings` instance with the same value, positional, and contextual information as the original.
        """
        copy = Strings(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy
    
    def concat(self, other):
        """
        Concatenates the current string with another `Strings` instance.

        Args:
            other (Strings): The `Strings` instance to concatenate with.

        Returns:
            Strings: A new `Strings` instance with the concatenated value, or None if `other` is not of type `Strings`.
        """
        if isinstance(other, Strings):
            return Strings(self.value + other.value).set_context(self.context)
        return None
    
    def __repr__(self):
        """
        Returns a string representation of the string value.

        Returns:
            str: The string value.
        """
        return self.value

class CompOp:
    """
    A class representing a comparison operation.

    This class encapsulates a comparison value and provides methods for performing various comparison operations.
    It also supports setting positional and contextual information and converting the comparison result to a string or boolean.

    Attributes:
        value (any): The comparison value (e.g., boolean or result of a comparison operation).
        pos_start (Position or None): The start position of the token in the source code.
        pos_end (Position or None): The end position of the token in the source code.
        context (Context or None): The context in which this comparison operation is used.

    Methods:
        set_pos(pos_start=None, pos_end=None):
            Sets the positional information of the comparison operation.

        set_context(context=None):
            Sets the context of the comparison operation.

        not_this():
            Performs a logical NOT operation on the comparison value and returns the result as a new `CompOp` instance.

        equal_to(other):
            Compares the current comparison value for equality with another `CompOp` instance's value.
            Returns a new `CompOp` instance with the result of the comparison, or None if the other instance is not of type `CompOp`.

        not_equal_to(other):
            Compares the current comparison value for inequality with another `CompOp` instance's value.
            Returns a new `CompOp` instance with the result of the comparison, or None if the other instance is not of type `CompOp`.

        greater_than(other):
            Compares the current comparison value to check if it is greater than another `CompOp` instance's value.
            Returns a new `CompOp` instance with the result of the comparison, or None if the other instance is not of type `CompOp`.

        less_than(other):
            Compares the current comparison value to check if it is less than another `CompOp` instance's value.
            Returns a new `CompOp` instance with the result of the comparison, or None if the other instance is not of type `CompOp`.

        greater_than_or_equal_to(other):
            Compares the current comparison value to check if it is greater than or equal to another `CompOp` instance's value.
            Returns a new `CompOp` instance with the result of the comparison, or None if the other instance is not of type `CompOp`.

        less_than_or_equal_to(other):
            Compares the current comparison value to check if it is less than or equal to another `CompOp` instance's value.
            Returns a new `CompOp` instance with the result of the comparison, or None if the other instance is not of type `CompOp`.

        __str__():
            Returns a string representation of the comparison value.

        __bool__():
            Returns a boolean representation of the comparison value.
    """

    def __init__(self, value):
        """
        Initializes a new `CompOp` instance with the given `value`.

        Args:
            value (any): The comparison value.
        """
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        """
        Sets the positional information for the comparison operation.

        Args:
            pos_start (Position or None): The start position of the token.
            pos_end (Position or None): The end position of the token.

        Returns:
            CompOp: The current `CompOp` instance with updated positional information.
        """
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        """
        Sets the context for the comparison operation.

        Args:
            context (Context or None): The context in which this comparison operation is used.

        Returns:
            CompOp: The current `CompOp` instance with updated context.
        """
        self.context = context
        return self
    
    def not_this(self):
        """
        Performs a logical NOT operation on the comparison value.

        Returns:
            CompOp: A new `CompOp` instance with the result of the logical NOT operation.
        """
        return CompOp(not self.value).set_context(self.context), None
    
    def equal_to(self, other):
        """
        Compares the current comparison value for equality with another `CompOp` instance's value.

        Args:
            other (CompOp): The `CompOp` instance to compare with.

        Returns:
            CompOp: A new `CompOp` instance with the result of the equality comparison, or None if `other` is not of type `CompOp`.
        """
        if isinstance(other, CompOp):
            return CompOp(self.value == other.value).set_context(self.context), None
        
    def not_equal_to(self, other):
        """
        Compares the current comparison value for inequality with another `CompOp` instance's value.

        Args:
            other (CompOp): The `CompOp` instance to compare with.

        Returns:
            CompOp: A new `CompOp` instance with the result of the inequality comparison, or None if `other` is not of type `CompOp`.
        """
        if isinstance(other, CompOp):
            return CompOp(self.value != other.value).set_context(self.context), None

    def greater_than(self, other):
        """
        Compares the current comparison value to check if it is greater than another `CompOp` instance's value.

        Args:
            other (CompOp): The `CompOp` instance to compare with.

        Returns:
            CompOp: A new `CompOp` instance with the result of the greater-than comparison, or None if `other` is not of type `CompOp`.
        """
        if isinstance(other, CompOp):
            return CompOp(self.value > other.value).set_context(self.context), None

    def less_than(self, other):
        """
        Compares the current comparison value to check if it is less than another `CompOp` instance's value.

        Args:
            other (CompOp): The `CompOp` instance to compare with.

        Returns:
            CompOp: A new `CompOp` instance with the result of the less-than comparison, or None if `other` is not of type `CompOp`.
        """
        if isinstance(other, CompOp):
            return CompOp(self.value < other.value).set_context(self.context), None

    def greater_than_or_equal_to(self, other):
        """
        Compares the current comparison value to check if it is greater than or equal to another `CompOp` instance's value.

        Args:
            other (CompOp): The `CompOp` instance to compare with.

        Returns:
            CompOp: A new `CompOp` instance with the result of the greater-than-or-equal-to comparison, or None if `other` is not of type `CompOp`.
        """
        if isinstance(other, CompOp):
            return CompOp(self.value >= other.value).set_context(self.context), None

    def less_than_or_equal_to(self, other):
        """
        Compares the current comparison value to check if it is less than or equal to another `CompOp` instance's value.

        Args:
            other (CompOp): The `CompOp` instance to compare with.

        Returns:
            CompOp: A new `CompOp` instance with the result of the less-than-or-equal-to comparison, or None if `other` is not of type `CompOp`.
        """
        if isinstance(other, CompOp):
            return CompOp(self.value <= other.value).set_context(self.context), None

    def __str__(self):
        """
        Returns a string representation of the comparison value.

        Returns:
            str: The comparison value as a string.
        """
        return str(self.value)
    
    def __bool__(self):
        """
        Returns a boolean representation of the comparison value.

        Returns:
            bool: The boolean value of the comparison.
        """
        return bool(self.value)

class LogicalOp:
    """
    A class representing a logical operation.

    This class encapsulates a logical value (True or False) and provides methods for performing logical operations like NOT, AND, and OR.

    Attributes:
        value (bool): The logical value of the operation.
        pos_start (Position or None): The start position of the token in the source code.
        pos_end (Position or None): The end position of the token in the source code.
        context (Context or None): The context in which this logical operation is used.

    Methods:
        set_pos(pos_start=None, pos_end=None):
            Sets the positional information of the logical operation.

        set_context(context=None):
            Sets the context of the logical operation.

        not_this():
            Performs a logical NOT operation on the logical value and returns the result as a new `LogicalOp` instance.

        and_with(other):
            Performs a logical AND operation with another `LogicalOp` or `Number` instance.
            Returns a new `LogicalOp` instance with the result of the AND operation, or None if `other` is not of type `LogicalOp` or `Number`.

        or_with(other):
            Performs a logical OR operation with another `LogicalOp` or `Number` instance.
            Returns a new `LogicalOp` instance with the result of the OR operation, or None if `other` is not of type `LogicalOp` or `Number`.

        __str__():
            Returns a string representation of the logical value.
    """

    def __init__(self, value):
        """
        Initializes a new `LogicalOp` instance with the given `value`.

        Args:
            value (any): The logical value (True or False) or something that can be converted to a boolean.
        """
        self.value = bool(value)
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        """
        Sets the positional information for the logical operation.

        Args:
            pos_start (Position or None): The start position of the token.
            pos_end (Position or None): The end position of the token.

        Returns:
            LogicalOp: The current `LogicalOp` instance with updated positional information.
        """
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        """
        Sets the context for the logical operation.

        Args:
            context (Context or None): The context in which this logical operation is used.

        Returns:
            LogicalOp: The current `LogicalOp` instance with updated context.
        """
        self.context = context
        return self
    
    def not_this(self):
        """
        Performs a logical NOT operation on the logical value.

        Returns:
            LogicalOp: A new `LogicalOp` instance with the result of the logical NOT operation.
        """
        return LogicalOp(not self.value).set_context(self.context), None 

    def and_with(self, other):
        """
        Performs a logical AND operation with another `LogicalOp` or `Number` instance.

        Args:
            other (LogicalOp or Number): The `LogicalOp` or `Number` instance to perform the AND operation with.

        Returns:
            LogicalOp: A new `LogicalOp` instance with the result of the logical AND operation, or None if `other` is not of type `LogicalOp` or `Number`.
        """
        if isinstance(other, (LogicalOp, Number)):
            return LogicalOp(self.value and bool(other.value)).set_context(self.context), None

    def or_with(self, other):
        """
        Performs a logical OR operation with another `LogicalOp` or `Number` instance.

        Args:
            other (LogicalOp or Number): The `LogicalOp` or `Number` instance to perform the OR operation with.

        Returns:
            LogicalOp: A new `LogicalOp` instance with the result of the logical OR operation, or None if `other` is not of type `LogicalOp` or `Number`.
        """
        if isinstance(other, (LogicalOp, Number)):
            return LogicalOp(self.value or bool(other.value)).set_context(self.context), None

    def __str__(self):
        """
        Returns a string representation of the logical value.

        Returns:
            str: The logical value as a string.
        """
        return str(self.value)


class Variable:
    """
    A class representing a variable storage.

    This class allows assignment of `Number` values to variable names and retrieval of their values. It supports basic operations for setting and getting variable values.

    Attributes:
        value (any): The value assigned to the variable (if any).
        pos_start (Position or None): The start position of the token in the source code.
        pos_end (Position or None): The end position of the token in the source code.
        context (Context or None): The context in which this variable is used.
        variables (dict): A dictionary storing variable names and their corresponding `Number` instances.

    Methods:
        set_pos(pos_start=None, pos_end=None):
            Sets the positional information of the variable.

        set_context(context=None):
            Sets the context of the variable.

        assign_to(name, other):
            Assigns a `Number` value to a variable name.
            Returns the assigned `Number` instance and None, or None and an error if `other` is not a `Number`.

        get_value(name):
            Retrieves the value of a variable by its name.
            Returns the `Number` instance associated with the variable name, or None if the variable does not exist.
    """

    def __init__(self, value=None):
        """
        Initializes a new `Variable` instance.

        Args:
            value (any): The initial value of the variable (if any).
        """
        self.set_pos()
        self.set_context()
        self.value = value
        self.variables = {}  

    def set_pos(self, pos_start=None, pos_end=None):
        """
        Sets the positional information for the variable.

        Args:
            pos_start (Position or None): The start position of the token.
            pos_end (Position or None): The end position of the token.

        Returns:
            Variable: The current `Variable` instance with updated positional information.
        """
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        """
        Sets the context for the variable.

        Args:
            context (Context or None): The context in which this variable is used.

        Returns:
            Variable: The current `Variable` instance with updated context.
        """
        self.context = context
        return self

    def assign_to(self, name, other):
        """
        Assigns a `Number` value to a variable name.

        Args:
            name (str): The name of the variable to assign the value to.
            other (Number): The `Number` instance to assign to the variable.

        Returns:
            Number: The assigned `Number` instance, or None if `other` is not a `Number`.
            ValueError: An error indicating that assignment only supports `Number` instances.
        """
        if isinstance(other, Number):
            self.variables[name] = Number(other.value).set_context(self.context)
            return self.variables[name], None
        else:
            return None, ValueError("Assignment only supports Numbers.")

    def get_value(self, name):
        """
        Retrieves the value of a variable by its name.

        Args:
            name (str): The name of the variable whose value is to be retrieved.

        Returns:
            Number or None: The `Number` instance associated with the variable name, or None if the variable does not exist.
        """
        return self.variables.get(name, None)
    
#################################
# CONTEXT
#################################
    
class Context:
    """
    A class representing the context in which code execution takes place.

    This class encapsulates the environment in which variables are stored and code is executed. It manages variable storage and allows for nested contexts, where a context can have a parent context.

    Attributes:
        display_name (str): The name of this context, used for display purposes or debugging.
        parent (Context or None): The parent context of this context, or None if this is the root context.
        parent_entry_pos (Position or None): The position in the source code where this context was entered, or None if not applicable.
        variables (dict): A dictionary storing variable names and their corresponding values in this context.

    Methods:
        __init__(display_name, parent=None, parent_entry_pos=None):
            Initializes a new `Context` instance with the given display name, parent context, and parent entry position.
    """

    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        """
        Initializes a new `Context` instance.

        Args:
            display_name (str): The name of this context, used for display purposes or debugging.
            parent (Context or None, optional): The parent context of this context. Defaults to None.
            parent_entry_pos (Position or None, optional): The position in the source code where this context was entered. Defaults to None.
        """
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.variables = {}

#################################
# INTERPRETER
#################################
    
class Interpreter:
    """
    A class that interprets AST (Abstract Syntax Tree) nodes and executes corresponding operations.

    This class is responsible for visiting nodes in an AST and performing operations based on the node types. It handles arithmetic, logical operations, comparisons, variable assignments, and control flow constructs.

    Methods:
        visit(node, context):
            Visits a node based on its type and executes the corresponding method.

        no_visit_method(node, context):
            A fallback method called when no specific visit method is defined for the node's type.

        visit_NumberNode(node, context):
            Handles `NumberNode` instances, returning the value as a `Number` object.

        visit_StringNode(node, context):
            Handles `StringNode` instances, returning the value as a `Strings` object.

        visit_BinOpNode(node, context):
            Handles `BinOpNode` instances, performing binary operations (addition, subtraction, multiplication, division, power) on `Number` or `Strings` objects.

        visit_UnaryOpNode(node, context):
            Handles `UnaryOpNode` instances, performing unary operations (negation) on `Number` objects.

        visit_MinNode(node, context):
            Handles `MinNode` instances, returning the minimum value between two `Number` nodes.

        visit_MaxNode(node, context):
            Handles `MaxNode` instances, returning the maximum value between two `Number` nodes.

        visit_UnaryLogicalOpNode(node, context):
            Handles `UnaryLogicalOpNode` instances, performing logical negation on `LogicalOp` or `Number` objects.

        visit_LogicalOpNode(node, context):
            Handles `LogicalOpNode` instances, performing logical operations (AND, OR) on `LogicalOp` or `Number` objects.

        visit_CompOpNode(node, context):
            Handles `CompOpNode` instances, performing comparison operations (greater than, less than, equal to, etc.) on `Number` objects.

        visit_StatementsNode(node, context):
            Handles `StatementsNode` instances, executing a list of statements and returning their values.

        visit_VariableNode(node, context):
            Handles `VariableNode` instances, performing variable assignment or access.

        visit_IfNode(node, context):
            Handles `IfNode` instances, executing the body of the first true condition or an `ELSE` branch if present.

        visit_WhileNode(node, context):
            Handles `WhileNode` instances, executing the loop body while the condition is true and returning the final value of variable `x`.

        visit_PrintNode(node, context):
            Handles `PrintNode` instances, printing the value of the node to the standard output.
    """

    def visit(self, node, context):
        """
        Visits a node based on its type and executes the corresponding method.

        Args:
            node: The AST node to visit.
            context: The context in which the node is evaluated.

        Returns:
            The result of visiting the node.
        """
        method_name = f'visit_{type(node).__name__}' 
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)
    
    def no_visit_method(self, node, context):
        """
        A fallback method called when no specific visit method is defined for the node's type.

        Args:
            node: The AST node to visit.
            context: The context in which the node is evaluated.

        Raises:
            Exception: If no visit method is defined for the node's type.
        """
        raise Exception(f'No visit_{type(node).__name__} method defined')
    
    def visit_NumberNode(self, node, context):
        """
        Handles `NumberNode` instances, returning the value as a `Number` object.

        Args:
            node: The `NumberNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the `Number` object or an error.
        """
        return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))
    
    def visit_StringNode(self, node, context):
        """
        Handles `StringNode` instances, returning the value as a `Strings` object.

        Args:
            node: The `StringNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the `Strings` object or an error.
        """
        return RTResult().success(Strings(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))

    def visit_BinOpNode(self, node, context):
        """
        Handles `BinOpNode` instances, performing binary operations (addition, subtraction, multiplication, division, power) on `Number` or `Strings` objects.

        Args:
            node: The `BinOpNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the result of the binary operation or an error.
        """
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        error = None  

        # Handle numbers
        if isinstance(left, Number) and isinstance(right, Number):
            if node.op_tok.type == TT_PLUS:
                result, error = left.added_to(right)
            elif node.op_tok.type == TT_MINUS:
                result, error = left.subbed_by(right)
            elif node.op_tok.type == TT_MUL:
                result, error = left.multed_by(right)
            elif node.op_tok.type == TT_DIV:
                result, error = left.dived_by(right)
            elif node.op_tok.type == TT_MODULO:
                result, error = left.modulo(right)
            elif node.op_tok.type == TT_POWER:
                result, error = left.powed_by(right)

        # Handle strings
        elif isinstance(left, Strings) and isinstance(right, Strings):
            if node.op_tok.type == TT_PLUS:  
                result = left.concat(right)
                if result is None:
                    error = ValueError(f"Cannot concatenate {left} and {right}")
            else:
                error = ValueError(f"Unsupported operation for strings: {node.op_tok}")

        if error:
            return res.failure(error)
        elif result is not None:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        """
        Handles `UnaryOpNode` instances, performing unary operations (negation) on `Number` objects.

        Args:
            node: The `UnaryOpNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the result of the unary operation or an error.
        """
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error:
            return res

        error = None  # Initialize error to None

        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_by(Number(-1))

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_MinNode(self, node, context):
        """
        Handles `MinNode` instances, returning the minimum value between two `Number` nodes.

        Args:
            node: The `MinNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the `Number` object representing the minimum value or an error.
        """
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        min_value = min(left.value, right.value)
        return res.success(Number(min_value).set_context(context).set_pos(node.pos_start, node.pos_end))
    
    def visit_MaxNode(self, node, context):
        """
        Handles `MaxNode` instances, returning the maximum value between two `Number` nodes.

        Args:
            node: The `MaxNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the `Number` object representing the maximum value or an error.
        """
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        max_value = max(left.value, right.value)
        return res.success(Number(max_value).set_context(context).set_pos(node.pos_start, node.pos_end))
    
    def visit_UnaryLogicalOpNode(self, node, context):
        """
        Handles `UnaryLogicalOpNode` instances, performing logical negation on `LogicalOp` or `Number` objects.

        Args:
            node: The `UnaryLogicalOpNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the result of the logical negation or an error.
        """
        res = RTResult()
        operand = res.register(self.visit(node.node, context))
        if res.error:
            return res

        if node.op_tok.type == TT_NOT:
            if hasattr(operand, 'not_this'):
                result, error = operand.not_this()
            else:
                result, error = LogicalOp(operand.value).not_this()
            
            if error:
                return res.failure(error)
            return res.success(result.set_pos(node.pos_start, node.pos_end))
        
        return res.failure(InvalidSyntaxError(
            node.pos_start, node.pos_end,
            f"Invalid unary operator '{node.op_tok.type}'"
        ))
        
    def visit_LogicalOpNode(self, node, context):
        """
        Handles `LogicalOpNode` instances, performing logical operations (AND, OR) on `LogicalOp` or `Number` objects.

        Args:
            node: The `LogicalOpNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the result of the logical operation or an error.
        """
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: 
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: 
            return res

        if node.op_tok.type == TT_AND:
            result, error = LogicalOp(left.value).and_with(LogicalOp(right.value))
        elif node.op_tok.type == TT_OR:
            result, error = LogicalOp(left.value).or_with(LogicalOp(right.value))
        else:
            return res.failure(InvalidSyntaxError(
                node.pos_start, node.pos_end,
                f"Invalid logical operator '{node.op_tok.type}'"
            ))

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))
        
    def visit_CompOpNode(self, node, context):
        """
        Handles `CompOpNode` instances, performing comparison operations (greater than, less than, equal to, etc.) on `Number` objects.

        Args:
            node: The `CompOpNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the result of the comparison operation or an error.
        """
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        if isinstance(left, Number) and isinstance(right, Number):
            left = CompOp(left.value).set_context(context).set_pos(node.pos_start, node.pos_end)
            right = CompOp(right.value).set_context(context).set_pos(node.pos_start, node.pos_end)

        if node.op_tok.type == TT_GT:
            result, error = left.greater_than(right)
        elif node.op_tok.type == TT_LT:
            result, error = left.less_than(right)
        elif node.op_tok.type == TT_GTE:
            result, error = left.greater_than_or_equal_to(right)
        elif node.op_tok.type == TT_LTE:
            result, error = left.less_than_or_equal_to(right)
        elif node.op_tok.type == TT_EQUAL:
            result, error = left.equal_to(right)
        elif node.op_tok.type == TT_NEQUAL:
            result, error = left.not_equal_to(right)
        
        if error:
            return res.failure(error)
        elif result is not None:
            return res.success(result.set_pos(node.pos_start, node.pos_end))
        else:
            return res.failure(InvalidSyntaxError(
                node.pos_start, node.pos_end,
                f"Operation failed unexpectedly with operator '{node.op_tok.type}'\n"
            ))
        
    def visit_StatementsNode(self, node, context):
        """
        Handles `StatementsNode` instances, executing a list of statements and returning their values.

        Args:
            node: The `StatementsNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the list of values resulting from the executed statements or an error.
        """
        res = RTResult()
        results = []  # Just for debugging
        for statement in node.statements:
            value = res.register(self.visit(statement, context))
            if res.error:
                return res

            # If the value is an instance of a custom class like Strings or Number, get its value
            if isinstance(value, Strings):
                value = value.value
            elif isinstance(value, Number):
                value = value.value
            
            results.append(value)
        return res.success(results)

    def visit_VariableNode(self, node, context):
        """
        Handles `VariableNode` instances, performing variable assignment or access.

        Args:
            node: The `VariableNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the assigned value or the value of the variable or an error.
        """
        res = RTResult()

        # Get the variable name directly from the token
        variable_name = node.var_name.value

        if node.value:
            # Variable assignment
            value = res.register(self.visit(node.value, context))
            if res.error:
                return res
            context.variables[variable_name] = value
            return res.success(value)
        else:
            # Variable access
            if variable_name in context.variables:
                value = context.variables[variable_name]
                # If the value is a Number or Strings, return a copy
                if isinstance(value, (Number, Strings)):
                    return res.success(value.copy().set_pos(node.pos_start, node.pos_end))
                return res.success(value)
            else:
                return res.failure(RTError(
                    node.pos_start, node.pos_end,
                    f"'{variable_name}' is not defined",
                    context
                ))
            
    def visit_IfNode(self, node, context):
        """
        Handles `IfNode` instances, executing the body of the first true condition or an `ELSE` branch if present.

        Args:
            node: The `IfNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the result of the executed body or `None` if no condition was true.
        """
        res = RTResult()
        executed = False

        for i, (condition, body) in enumerate(node.cases):
            if executed:
                break  

            if condition is None:
                body_result = self.visit(body, context)
                if body_result.error:
                    return res.failure(body_result.error)
                executed = True
                return res.success(body_result.value)

            condition_result = self.visit(condition, context)
            if condition_result.error:
                return res.failure(condition_result.error)

            if condition_result.value:
                body_result = self.visit(body, context)
                if body_result.error:
                    return res.failure(body_result.error)
                executed = True
                return res.success(body_result.value)

        return res.success(None)

    def visit_WhileNode(self, node, context):
        """
        Handles `WhileNode` instances, executing the loop body while the condition is true and returning the final value of variable `x`.

        Args:
            node: The `WhileNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with the final value of variable `x` or an error.
        """
        res = RTResult()
        
        while True:
            condition = res.register(self.visit(node.condition, context))
            if res.should_return(): 
                return res
            
            if isinstance(condition, tuple) and len(condition) == 2:
                condition = condition[0]
            
            if not bool(condition):
                break

            for statement in node.body:
                res.register(self.visit(statement, context))
                if res.should_return(): 
                    return res

        # Return the final value of x
        x_value = context.variables.get('x')
        return res.success(x_value)
    
    def visit_PrintNode(self, node, context):
        """
        Handles `PrintNode` instances, printing the value of the node to the standard output.

        Args:
            node: The `PrintNode` instance to visit.
            context: The context in which the node is evaluated.

        Returns:
            An `RTResult` instance with `None` indicating successful printing or an error.
        """
        res = RTResult()
        value = res.register(self.visit(node.node_to_print, context))
        if res.error:
            return res
        print(value)
        return res.success(None)

#################################
# RUN
#################################
        
def run(filename, text):
    # Generate tokens
    lexer = Lexer(filename, text)
    tokens, error = lexer.make_tokens()
    if error: 
        return None, error
    
    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: 
        return (None, ast.error)

    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    result = interpreter.visit(ast.node, context)
    
    if result.error:
        return None, result.error
    
    return result.value, None
