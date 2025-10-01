"""
LangChain Service Layer for Football Analytics
Enhanced ML Query Pipeline with Natural Language Processing
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory

from footballviz.query_builder import FilterCondition, FilterOperator, LogicOperator


class FootballAnalyticsCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for logging LangChain operations"""
    
    def __init__(self):
        self.logs = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logs.append({
            "type": "llm_start",
            "timestamp": datetime.now().isoformat(),
            "prompts": prompts
        })
    
    def on_llm_end(self, response, **kwargs):
        self.logs.append({
            "type": "llm_end", 
            "timestamp": datetime.now().isoformat(),
            "response": str(response)
        })


class FootballLangChainService:
    """Enhanced AI service using LangChain for football analytics"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url
        self.model = model
        self.callback_handler = FootballAnalyticsCallbackHandler()
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            base_url=base_url,
            model=model,
            temperature=0.3,
            num_predict=500,
            callbacks=[self.callback_handler]
        )
        
        # Initialize memory for conversations
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        # We'll handle filtering directly since we work with extracted data
        # No need for CustomQueryBuilder which requires database session
        
        self._setup_prompt_templates()
    
    def _setup_prompt_templates(self):
        """Setup reusable prompt templates"""
        
        # Natural Language to SQL Query Template
        self.nl_to_sql_template = ChatPromptTemplate.from_messages([
            ("system", """You are a football analytics expert. Convert natural language queries to structured filters for football game data.

AVAILABLE FIELDS:
- down: Integer (1-4), the down number
- distance: Integer, yards needed for first down
- yard_line: Integer (1-100), field position
- formation: String, offensive formation (e.g., "I-Formation", "Shotgun", "Singleback")
- play_type: String, type of play (e.g., "Run", "Pass", "Punt", "Field Goal")
- play_name: String, specific play call
- result_of_play: String, outcome description
- yards_gained: Integer, yards gained/lost on play

OPERATORS: equals, not_equals, greater_than, less_than, between, in, contains

Convert the query to this JSON format:
{{
  "conditions": [
    {{
      "field": "field_name",
      "operator": "operator_name", 
      "value": "value_or_array"
    }}
  ],
  "logic": "and"  // or "or"
}}

Examples:
"Red zone plays" -> {{"conditions": [{{"field": "yard_line", "operator": "between", "value": [1, 20]}}], "logic": "and"}}
"Third down conversions" -> {{"conditions": [{{"field": "down", "operator": "equals", "value": 3}}], "logic": "and"}}
"Running plays for more than 5 yards" -> {{"conditions": [{{"field": "play_type", "operator": "equals", "value": "Run"}}, {{"field": "yards_gained", "operator": "greater_than", "value": 5}}], "logic": "and"}}"""),
            ("user", "{query}")
        ])
        
        # Football Data Analysis Template  
        self.analysis_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert football analytics consultant. Analyze game data and provide specific, actionable insights.

Focus on:
- Trends and patterns in the data
- Strengths and weaknesses identification
- Strategic recommendations
- Specific statistics and comparisons
- Context about football strategy

Be specific with numbers and percentages. Provide clear, actionable advice."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", """
GAME DATA ANALYSIS REQUEST:
Query: {query}

DATA SUMMARY:
{data_summary}

TOP FORMATIONS:
{formations}

TOP PLAY TYPES: 
{play_types}

SITUATIONAL BREAKDOWNS:
{situations}

Provide a detailed analysis answering the user's question with specific insights and recommendations.
""")
        ])
        
        # Multi-step Analysis Template
        self.multi_step_template = ChatPromptTemplate.from_messages([
            ("system", """You are conducting a multi-step football analysis. Each step builds on previous findings.

Current step: {step}
Previous results: {previous_results}

Analyze the current data in context of previous findings and provide insights for this step."""),
            ("user", "{current_query}")
        ])
    
    def is_available(self) -> bool:
        """Check if the LangChain service is available"""
        try:
            test_response = self.llm.invoke("test")
            return bool(test_response)
        except Exception:
            return False
    
    def natural_language_to_sql(self, query: str) -> Dict[str, Any]:
        """Convert natural language query to SQL filter conditions"""
        try:
            chain = self.nl_to_sql_template | self.llm | JsonOutputParser()
            result = chain.invoke({"query": query})
            return result
        except Exception as e:
            logging.error(f"Error in natural language to SQL conversion: {str(e)}")
            # Fallback to simple keyword matching
            return self._fallback_query_parsing(query)
    
    def _fallback_query_parsing(self, query: str) -> Dict[str, Any]:
        """Simple keyword-based query parsing as fallback"""
        query_lower = query.lower()
        conditions = []
        
        # Common patterns
        if "red zone" in query_lower:
            conditions.append({"field": "yard_line", "operator": "between", "value": [1, 20]})
        if "third down" in query_lower:
            conditions.append({"field": "down", "operator": "equals", "value": 3})
        if "fourth down" in query_lower:
            conditions.append({"field": "down", "operator": "equals", "value": 4})
        if "running" in query_lower or "run" in query_lower:
            conditions.append({"field": "play_type", "operator": "equals", "value": "Run"})
        if "passing" in query_lower or "pass" in query_lower:
            conditions.append({"field": "play_type", "operator": "equals", "value": "Pass"})
        
        return {"conditions": conditions, "logic": "and"}
    
    def analyze_football_data_enhanced(self, query: str, plays_data: List[Dict]) -> str:
        """Enhanced football data analysis with LangChain"""
        if not plays_data:
            return "No game data available to analyze."
        
        # Generate data summary
        data_summary = self._generate_data_summary(plays_data)
        formations = self._analyze_formations(plays_data)
        play_types = self._analyze_play_types(plays_data)
        situations = self._analyze_situations(plays_data)
        
        # Store conversation context
        self.memory.chat_memory.add_user_message(query)
        
        # Create analysis chain
        chain = self.analysis_template | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "query": query,
            "data_summary": data_summary,
            "formations": formations,
            "play_types": play_types,
            "situations": situations,
            "chat_history": self.memory.chat_memory.messages
        })
        
        # Store AI response in memory
        self.memory.chat_memory.add_ai_message(response)
        
        return response
    
    def multi_step_analysis(self, queries: List[str], plays_data: List[Dict]) -> List[Dict[str, Any]]:
        """Perform multi-step analysis with context preservation"""
        results = []
        previous_results = []
        
        for i, query in enumerate(queries):
            try:
                # For first step, use regular analysis
                if i == 0:
                    response = self.analyze_football_data_enhanced(query, plays_data)
                else:
                    # For subsequent steps, use multi-step template
                    chain = self.multi_step_template | self.llm | StrOutputParser()
                    response = chain.invoke({
                        "step": i + 1,
                        "current_query": query,
                        "previous_results": "\n".join([f"Step {j+1}: {r['response']}" for j, r in enumerate(previous_results)])
                    })
                
                result = {
                    "step": i + 1,
                    "query": query,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                previous_results.append(result)
                
            except Exception as e:
                logging.error(f"Error in multi-step analysis step {i+1}: {str(e)}")
                results.append({
                    "step": i + 1,
                    "query": query,
                    "response": f"Error processing step: {str(e)}",
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def natural_language_to_sql(self, query: str) -> Dict[str, Any]:
        """Convert natural language query to SQL-like filters"""
        try:
            # Simple pattern matching for common queries
            query_lower = query.lower()
            filters = {"conditions": []}
            
            # Red zone detection
            if "red zone" in query_lower:
                filters["conditions"].append({
                    "field": "yard_line",
                    "operator": "<=",
                    "value": 20
                })
            
            # Third down detection
            if "third down" in query_lower:
                filters["conditions"].append({
                    "field": "down",
                    "operator": "=",
                    "value": 3
                })
            
            # Fourth down detection
            if "fourth down" in query_lower:
                filters["conditions"].append({
                    "field": "down",
                    "operator": "=",
                    "value": 4
                })
            
            # Formation detection
            if "shotgun" in query_lower:
                filters["conditions"].append({
                    "field": "formation",
                    "operator": "contains",
                    "value": "Shotgun"
                })
            
            # Big plays detection
            if "big play" in query_lower or "over 15" in query_lower:
                filters["conditions"].append({
                    "field": "yards_gained",
                    "operator": ">",
                    "value": 15
                })
            
            # Running plays over X yards
            if "running" in query_lower and ("over" in query_lower or "more than" in query_lower):
                if "5" in query_lower:
                    filters["conditions"].append({
                        "field": "play_type",
                        "operator": "=",
                        "value": "Run"
                    })
                    filters["conditions"].append({
                        "field": "yards_gained",
                        "operator": ">",
                        "value": 5
                    })
            
            return filters
            
        except Exception as e:
            logging.error(f"Error in natural_language_to_sql: {str(e)}")
            return {"conditions": []}

    def conversational_query(self, query: str, plays_data: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process conversational queries with memory"""
        # Check if this might be a natural language filter query
        if any(keyword in query.lower() for keyword in ["show", "find", "get", "filter", "where"]):
            # Try to convert to SQL first
            try:
                sql_filters = self.natural_language_to_sql(query)
                if sql_filters.get("conditions"):
                    # Apply filters to data
                    filtered_data = self._apply_filters_to_data(plays_data, sql_filters)
                    analysis = self.analyze_football_data_enhanced(query, filtered_data)
                    
                    return {
                        "type": "filtered_analysis",
                        "query": query,
                        "filters_applied": sql_filters,
                        "filtered_count": len(filtered_data),
                        "total_count": len(plays_data),
                        "analysis": analysis,
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                logging.warning(f"Failed to apply natural language filters: {str(e)}")
        
        # Regular conversational analysis
        analysis = self.analyze_football_data_enhanced(query, plays_data)
        
        return {
            "type": "conversational_analysis",
            "query": query,
            "analysis": analysis,
            "conversation_length": len(self.memory.chat_memory.messages),
            "timestamp": datetime.now().isoformat()
        }
    
    def _apply_filters_to_data(self, plays_data: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply parsed filters to plays data"""
        if not filters.get("conditions"):
            return plays_data
        
        filtered_data = []
        for play in plays_data:
            match = True
            
            for condition in filters["conditions"]:
                field = condition["field"]
                operator = condition["operator"]
                value = condition["value"]
                
                play_value = play.get(field)
                if play_value is None:
                    match = False
                    break
                
                # Apply filter logic
                if operator in ["equals", "="] and play_value != value:
                    match = False
                elif operator in ["greater_than", ">"] and play_value <= value:
                    match = False
                elif operator in ["less_than", "<"] and play_value >= value:
                    match = False
                elif operator in ["less_than_or_equal", "<="] and play_value > value:
                    match = False
                elif operator == "between" and not (value[0] <= play_value <= value[1]):
                    match = False
                elif operator == "contains" and str(value).lower() not in str(play_value).lower():
                    match = False
                
                if not match:
                    break
            
            if match:
                filtered_data.append(play)
        
        return filtered_data
    
    def _generate_data_summary(self, plays_data: List[Dict]) -> str:
        """Generate summary statistics from plays data"""
        total_plays = len(plays_data)
        total_yards = sum(play.get('yards_gained', 0) for play in plays_data)
        avg_yards = total_yards / total_plays if total_plays > 0 else 0
        
        successful_plays = len([p for p in plays_data if p.get('yards_gained', 0) > 0])
        success_rate = (successful_plays / total_plays * 100) if total_plays > 0 else 0
        
        return f"Total Plays: {total_plays}, Total Yards: {total_yards}, Avg Yards/Play: {avg_yards:.2f}, Success Rate: {success_rate:.1f}%"
    
    def _analyze_formations(self, plays_data: List[Dict]) -> str:
        """Analyze formation usage"""
        formations = {}
        for play in plays_data:
            formation = play.get('formation', 'Unknown')
            formations[formation] = formations.get(formation, 0) + 1
        
        top_formations = sorted(formations.items(), key=lambda x: x[1], reverse=True)[:5]
        return "\n".join([f"- {name}: {count} plays ({count/len(plays_data)*100:.1f}%)" 
                         for name, count in top_formations])
    
    def _analyze_play_types(self, plays_data: List[Dict]) -> str:
        """Analyze play type distribution"""
        play_types = {}
        for play in plays_data:
            play_type = play.get('play_type', 'Unknown')
            play_types[play_type] = play_types.get(play_type, 0) + 1
        
        top_types = sorted(play_types.items(), key=lambda x: x[1], reverse=True)[:5]
        return "\n".join([f"- {name}: {count} plays ({count/len(plays_data)*100:.1f}%)" 
                         for name, count in top_types])
    
    def _analyze_situations(self, plays_data: List[Dict]) -> str:
        """Analyze situational football data"""
        downs = {}
        distances = {"Short (1-3)": 0, "Medium (4-7)": 0, "Long (8+)": 0}
        field_position = {"Red Zone (1-20)": 0, "Mid Field (21-80)": 0, "Own End (81-100)": 0}
        
        for play in plays_data:
            # Down analysis
            down = play.get('down', 0)
            if down:
                downs[f"Down {down}"] = downs.get(f"Down {down}", 0) + 1
            
            # Distance analysis
            distance = play.get('distance', 0)
            if 1 <= distance <= 3:
                distances["Short (1-3)"] += 1
            elif 4 <= distance <= 7:
                distances["Medium (4-7)"] += 1
            elif distance >= 8:
                distances["Long (8+)"] += 1
            
            # Field position analysis
            yard_line = play.get('yard_line', 50)
            if 1 <= yard_line <= 20:
                field_position["Red Zone (1-20)"] += 1
            elif 21 <= yard_line <= 80:
                field_position["Mid Field (21-80)"] += 1
            else:
                field_position["Own End (81-100)"] += 1
        
        result = "DOWNS:\n"
        result += "\n".join([f"- {down}: {count} plays" for down, count in sorted(downs.items())])
        result += "\n\nDISTANCES:\n"
        result += "\n".join([f"- {dist}: {count} plays" for dist, count in distances.items()])
        result += "\n\nFIELD POSITION:\n"
        result += "\n".join([f"- {pos}: {count} plays" for pos, count in field_position.items()])
        
        return result
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        messages = []
        for message in self.memory.chat_memory.messages:
            messages.append({
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content,
                "timestamp": getattr(message, 'timestamp', datetime.now().isoformat())
            })
        return messages
    
    def clear_conversation_history(self):
        """Clear conversation memory"""
        self.memory.clear()
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics and health info"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "is_available": self.is_available(),
            "conversation_length": len(self.memory.chat_memory.messages),
            "callback_logs": len(self.callback_handler.logs),
            "last_activity": datetime.now().isoformat()
        }


# Initialize the enhanced service
langchain_service = FootballLangChainService()