"""
Natural Language Query Translator
Converts natural language queries to structured SQL filters using LangChain
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import logging
import re
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_ollama import OllamaLLM

from footballviz.query_builder import FilterCondition, FilterOperator, LogicOperator


@dataclass
class QueryTranslationResult:
    """Result of query translation"""
    success: bool
    filters: Optional[Dict[str, Any]] = None
    sql_conditions: Optional[List[FilterCondition]] = None
    error_message: Optional[str] = None
    confidence_score: float = 0.0
    suggested_corrections: Optional[List[str]] = None


class FootballQueryTranslator:
    """Translates natural language queries to SQL filters for football data"""
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        # We don't need the query builder since we work with extracted data
        self._setup_translation_templates()
        self._setup_field_mappings()
    
    def _setup_field_mappings(self):
        """Setup field mappings and validation"""
        self.field_info = {
            "down": {
                "type": "integer",
                "range": (1, 4),
                "description": "Down number (1st, 2nd, 3rd, 4th down)",
                "keywords": ["down", "first", "second", "third", "fourth", "1st", "2nd", "3rd", "4th"]
            },
            "distance": {
                "type": "integer", 
                "range": (1, 99),
                "description": "Yards needed for first down",
                "keywords": ["distance", "yards to go", "to go", "needed", "yardage"]
            },
            "yard_line": {
                "type": "integer",
                "range": (1, 100), 
                "description": "Field position (1=goal line, 50=midfield)",
                "keywords": ["yard line", "field position", "red zone", "goal line", "midfield"]
            },
            "formation": {
                "type": "string",
                "options": ["I-Formation", "Shotgun", "Singleback", "Pistol", "Wildcat", "Empty", "Pro Set"],
                "description": "Offensive formation",
                "keywords": ["formation", "i-formation", "shotgun", "singleback", "pistol", "wildcat", "empty"]
            },
            "play_type": {
                "type": "string", 
                "options": ["Run", "Pass", "Punt", "Field Goal", "Kickoff", "Extra Point"],
                "description": "Type of play called",
                "keywords": ["run", "running", "pass", "passing", "punt", "field goal", "kickoff", "extra point"]
            },
            "yards_gained": {
                "type": "integer",
                "range": (-20, 99),
                "description": "Yards gained or lost on the play", 
                "keywords": ["yards gained", "yards", "gain", "loss", "yardage", "positive", "negative"]
            }
        }
        
        self.common_patterns = {
            "red_zone": {"field": "yard_line", "operator": "between", "value": [1, 20]},
            "goal_line": {"field": "yard_line", "operator": "between", "value": [1, 5]},
            "midfield": {"field": "yard_line", "operator": "between", "value": [45, 55]},
            "short_yardage": {"field": "distance", "operator": "between", "value": [1, 3]},
            "long_yardage": {"field": "distance", "operator": "greater_than", "value": 7},
            "third_down": {"field": "down", "operator": "equals", "value": 3},
            "fourth_down": {"field": "down", "operator": "equals", "value": 4},
            "passing_down": {"field": "down", "operator": "in", "value": [2, 3]},
            "positive_plays": {"field": "yards_gained", "operator": "greater_than", "value": 0},
            "negative_plays": {"field": "yards_gained", "operator": "less_than", "value": 0},
            "big_plays": {"field": "yards_gained", "operator": "greater_than", "value": 15}
        }
    
    def _setup_translation_templates(self):
        """Setup LangChain prompt templates for query translation"""
        
        # Primary translation template
        self.translation_template = ChatPromptTemplate.from_messages([
            ("system", """You are a football analytics expert. Convert natural language queries about football data into structured JSON filters.

AVAILABLE FIELDS & OPERATORS:
- down: integer (1-4) | equals, not_equals, greater_than, less_than, in
- distance: integer (1-99) | equals, between, greater_than, less_than  
- yard_line: integer (1-100) | equals, between, greater_than, less_than
- formation: string | equals, contains, in
- play_type: string | equals, contains, in  
- yards_gained: integer (-20 to 99) | equals, between, greater_than, less_than

COMMON PATTERNS:
- "red zone" = yard_line between 1 and 20
- "third down" = down equals 3
- "passing plays" = play_type equals "Pass"
- "running plays" = play_type equals "Run"
- "short yardage" = distance between 1 and 3
- "big plays" = yards_gained greater_than 15

OUTPUT FORMAT (must be valid JSON):
{{
  "conditions": [
    {{
      "field": "field_name",
      "operator": "operator_name",
      "value": value_or_array
    }}
  ],
  "logic": "and",
  "confidence": 0.8,
  "interpretation": "Brief explanation of what you understood"
}}

IMPORTANT: 
- Only use fields and operators listed above
- Use "and" logic unless query explicitly mentions "or"
- Set confidence 0.0-1.0 based on query clarity
- If unsure, use most likely interpretation"""),
            ("user", "Query: {query}")
        ])
        
        # Validation template
        self.validation_template = ChatPromptTemplate.from_messages([
            ("system", """Validate and correct this football query filter if needed.

Check for:
1. Valid field names and operators
2. Reasonable value ranges  
3. Logical consistency
4. Football context accuracy

If corrections needed, provide improved version. Otherwise, return original."""),
            ("user", "Original filter: {filter_json}\nOriginal query: {original_query}")
        ])
    
    def translate_query(self, query: str) -> QueryTranslationResult:
        """Translate natural language query to SQL filters"""
        try:
            # Pre-process query
            processed_query = self._preprocess_query(query)
            
            # Check for common patterns first
            pattern_result = self._check_common_patterns(processed_query)
            if pattern_result:
                return pattern_result
            
            # Use LLM for complex translation
            llm_result = self._llm_translate(processed_query)
            if llm_result.success:
                # Validate and post-process
                validated_result = self._validate_translation(llm_result, query)
                return validated_result
            
            return llm_result
            
        except Exception as e:
            logging.error(f"Query translation error: {str(e)}")
            return QueryTranslationResult(
                success=False,
                error_message=f"Translation failed: {str(e)}",
                suggested_corrections=self._suggest_corrections(query)
            )
    
    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize the query"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Normalize common terms
        replacements = {
            "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
            "yds": "yards", "yd": "yard", "tds": "touchdowns", "td": "touchdown",
            "qb": "quarterback", "rb": "running back", "wr": "wide receiver"
        }
        
        for old, new in replacements.items():
            query = query.replace(old, new)
        
        return query
    
    def _check_common_patterns(self, query: str) -> Optional[QueryTranslationResult]:
        """Check for common football patterns first"""
        conditions = []
        confidence = 0.9
        interpretations = []
        
        # Check each pattern
        for pattern_name, pattern_filter in self.common_patterns.items():
            pattern_keywords = pattern_name.replace("_", " ")
            if pattern_keywords in query:
                conditions.append(pattern_filter.copy())
                interpretations.append(f"Detected {pattern_keywords}")
        
        # Specific field checks
        if "shotgun" in query:
            conditions.append({"field": "formation", "operator": "contains", "value": "Shotgun"})
            interpretations.append("Formation: Shotgun")
        
        if "run" in query and "pass" not in query:
            conditions.append({"field": "play_type", "operator": "equals", "value": "Run"})
            interpretations.append("Play type: Running")
        elif "pass" in query and "run" not in query:
            conditions.append({"field": "play_type", "operator": "equals", "value": "Pass"})
            interpretations.append("Play type: Passing")
        
        # Yardage patterns
        if "more than" in query or "greater than" in query:
            yards_match = re.search(r'(?:more than|greater than)\s+(\d+)\s*yards?', query)
            if yards_match:
                yards = int(yards_match.group(1))
                conditions.append({"field": "yards_gained", "operator": "greater_than", "value": yards})
                interpretations.append(f"Yards gained > {yards}")
        
        if conditions:
            filters = {
                "conditions": conditions,
                "logic": "and",
                "confidence": confidence,
                "interpretation": "; ".join(interpretations)
            }
            
            sql_conditions = self._convert_to_sql_conditions(conditions)
            
            return QueryTranslationResult(
                success=True,
                filters=filters,
                sql_conditions=sql_conditions,
                confidence_score=confidence
            )
        
        return None
    
    def _llm_translate(self, query: str) -> QueryTranslationResult:
        """Use LLM for complex query translation"""
        try:
            chain = self.translation_template | self.llm | JsonOutputParser()
            result = chain.invoke({"query": query})
            
            # Validate LLM response structure
            if not isinstance(result, dict) or "conditions" not in result:
                raise ValueError("Invalid LLM response structure")
            
            sql_conditions = self._convert_to_sql_conditions(result["conditions"])
            
            return QueryTranslationResult(
                success=True,
                filters=result,
                sql_conditions=sql_conditions,
                confidence_score=result.get("confidence", 0.5)
            )
            
        except Exception as e:
            logging.error(f"LLM translation error: {str(e)}")
            return QueryTranslationResult(
                success=False,
                error_message=f"LLM translation failed: {str(e)}"
            )
    
    def _validate_translation(self, result: QueryTranslationResult, original_query: str) -> QueryTranslationResult:
        """Validate and potentially correct the translation"""
        if not result.filters or not result.filters.get("conditions"):
            return result
        
        validated_conditions = []
        issues = []
        
        for condition in result.filters["conditions"]:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            # Validate field exists
            if field not in self.field_info:
                issues.append(f"Unknown field: {field}")
                continue
            
            # Validate operator
            valid_operators = ["equals", "not_equals", "greater_than", "less_than", "between", "in", "contains"]
            if operator not in valid_operators:
                issues.append(f"Invalid operator: {operator}")
                continue
            
            # Validate value range
            field_info = self.field_info[field]
            if field_info["type"] == "integer" and isinstance(value, (int, float)):
                field_range = field_info.get("range")
                if field_range and not (field_range[0] <= value <= field_range[1]):
                    issues.append(f"Value {value} out of range for {field}")
                    continue
            
            validated_conditions.append(condition)
        
        if issues:
            result.suggested_corrections = issues
            result.confidence_score *= 0.7  # Reduce confidence
        
        # Update conditions
        result.filters["conditions"] = validated_conditions
        result.sql_conditions = self._convert_to_sql_conditions(validated_conditions)
        
        return result
    
    def _convert_to_sql_conditions(self, conditions: List[Dict]) -> List[FilterCondition]:
        """Convert filter conditions to SQL FilterCondition objects"""
        sql_conditions = []
        
        for condition in conditions:
            try:
                field = condition["field"]
                operator_str = condition["operator"]
                value = condition["value"]
                
                # Map string operators to FilterOperator enum
                operator_mapping = {
                    "equals": FilterOperator.EQUALS,
                    "not_equals": FilterOperator.NOT_EQUALS,
                    "greater_than": FilterOperator.GREATER_THAN,
                    "less_than": FilterOperator.LESS_THAN,
                    "between": FilterOperator.BETWEEN,
                    "in": FilterOperator.IN,
                    "contains": FilterOperator.CONTAINS
                }
                
                operator = operator_mapping.get(operator_str)
                if operator:
                    sql_conditions.append(FilterCondition(field, operator, value))
                    
            except Exception as e:
                logging.warning(f"Failed to convert condition {condition}: {str(e)}")
        
        return sql_conditions
    
    def _suggest_corrections(self, query: str) -> List[str]:
        """Suggest corrections for failed queries"""
        suggestions = []
        
        # Common typos and suggestions
        if "yards" not in query and ("yard" in query or "yd" in query):
            suggestions.append("Did you mean 'yards' instead of 'yard' or 'yd'?")
        
        if re.search(r'\d+th', query):
            suggestions.append("Try using 'third down' or 'fourth down' instead of '3rd' or '4th'")
        
        if "endzone" in query:
            suggestions.append("Try 'red zone plays' or 'goal line plays'")
        
        # Suggest available fields
        suggestions.append("Available fields: down, distance, yard_line, formation, play_type, yards_gained")
        suggestions.append("Common queries: 'third down plays', 'red zone plays', 'running plays for more than 5 yards'")
        
        return suggestions
    
    def get_query_examples(self) -> Dict[str, str]:
        """Get example queries for documentation"""
        return {
            "Red zone plays": "Shows plays inside the 20-yard line",
            "Third down conversions": "Shows all third down attempts", 
            "Running plays for more than 5 yards": "Shows successful running plays",
            "Shotgun formation passes": "Shows passing plays from shotgun formation",
            "Fourth down plays": "Shows all fourth down attempts",
            "Big plays over 15 yards": "Shows plays with 15+ yard gains",
            "Short yardage situations": "Shows plays with 1-3 yards needed",
            "Goal line plays": "Shows plays inside the 5-yard line"
        }
    
    def analyze_query_difficulty(self, query: str) -> Dict[str, Any]:
        """Analyze how difficult a query is to translate"""
        words = query.lower().split()
        
        # Count football terms
        football_terms = 0
        for word in words:
            for field_info in self.field_info.values():
                if word in field_info.get("keywords", []):
                    football_terms += 1
                    break
        
        # Count common patterns
        pattern_matches = 0
        for pattern in self.common_patterns.keys():
            if pattern.replace("_", " ") in query.lower():
                pattern_matches += 1
        
        complexity_score = len(words) / 10 + (football_terms * 0.2) + (pattern_matches * 0.3)
        
        return {
            "word_count": len(words),
            "football_terms": football_terms,
            "pattern_matches": pattern_matches,
            "complexity_score": min(complexity_score, 1.0),
            "difficulty": "easy" if complexity_score < 0.3 else "medium" if complexity_score < 0.7 else "hard"
        }