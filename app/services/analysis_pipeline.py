"""
Multi-Step Analysis Pipeline for Football Analytics
Handles complex, sequential analysis workflows using LangChain
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM

from app.services.nl_query_translator import FootballQueryTranslator, QueryTranslationResult


class AnalysisStepType(Enum):
    """Types of analysis steps"""
    FILTER = "filter"
    AGGREGATE = "aggregate" 
    COMPARE = "compare"
    TREND = "trend"
    INSIGHT = "insight"
    RECOMMENDATION = "recommendation"


@dataclass
class AnalysisStep:
    """Individual analysis step"""
    step_id: str
    step_type: AnalysisStepType
    query: str
    description: str
    depends_on: List[str] = None
    filters: Optional[Dict] = None
    parameters: Optional[Dict] = None


@dataclass
class StepResult:
    """Result of an analysis step"""
    step_id: str
    step_type: AnalysisStepType
    success: bool
    data: Optional[Dict] = None
    insights: Optional[str] = None
    metrics: Optional[Dict] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    pipeline_id: str
    success: bool
    steps: List[StepResult]
    summary: Optional[str] = None
    recommendations: Optional[List[str]] = None
    total_execution_time: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class FootballAnalysisPipeline:
    """Multi-step football analysis pipeline using LangChain"""
    
    def __init__(self, llm: OllamaLLM, query_translator: FootballQueryTranslator):
        self.llm = llm
        self.query_translator = query_translator
        self._setup_analysis_templates()
        self._setup_predefined_workflows()
    
    def _setup_analysis_templates(self):
        """Setup prompt templates for different analysis types"""
        
        # Data filtering template
        self.filter_template = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing football data. Filter the provided dataset based on the query.

Current step: {step_description}
Previous results: {previous_results}

Analyze the filtered data and provide:
1. Summary of filtered results
2. Key statistics 
3. Notable patterns
4. Context for next analysis steps"""),
            ("user", "Query: {query}\nFiltered Data Count: {data_count}\nData Summary: {data_summary}")
        ])
        
        # Aggregation analysis template  
        self.aggregate_template = ChatPromptTemplate.from_messages([
            ("system", """You are performing aggregation analysis on football data.

Current step: {step_description}
Previous findings: {previous_results}

Calculate and analyze:
1. Key performance metrics
2. Averages, totals, percentages
3. Comparative analysis
4. Performance trends"""),
            ("user", "Data to analyze: {data_summary}\nSpecific focus: {query}")
        ])
        
        # Comparison analysis template
        self.comparison_template = ChatPromptTemplate.from_messages([
            ("system", """You are comparing different aspects of football performance.

Current step: {step_description}
Previous analysis: {previous_results}

Compare and contrast:
1. Performance differences
2. Strengths and weaknesses
3. Statistical significance
4. Strategic implications"""),
            ("user", "Comparison request: {query}\nData groups: {comparison_data}")
        ])
        
        # Trend analysis template
        self.trend_template = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing trends in football performance data.

Current step: {step_description}  
Historical context: {previous_results}

Identify and analyze:
1. Performance trends over time/situations
2. Improving or declining areas
3. Consistency patterns
4. Predictive indicators"""),
            ("user", "Trend analysis request: {query}\nTrend data: {trend_data}")
        ])
        
        # Insight generation template
        self.insight_template = ChatPromptTemplate.from_messages([
            ("system", """You are generating strategic insights from football analysis.

Analysis context: {step_description}
Previous findings: {previous_results}

Generate insights about:
1. Hidden patterns in the data
2. Strategic opportunities
3. Areas of concern
4. Competitive advantages"""),
            ("user", "Focus area: {query}\nAnalysis data: {analysis_data}")
        ])
        
        # Recommendation template
        self.recommendation_template = ChatPromptTemplate.from_messages([
            ("system", """You are a football strategy consultant providing actionable recommendations.

Analysis summary: {step_description}
Complete findings: {previous_results}

Provide specific recommendations for:
1. Strategic adjustments
2. Tactical improvements  
3. Player/position focus areas
4. Practice priorities"""),
            ("user", "Recommendation request: {query}\nFinal analysis: {final_data}")
        ])
        
        # Pipeline summary template
        self.summary_template = ChatPromptTemplate.from_messages([
            ("system", """Summarize the complete football analysis pipeline results.

Provide an executive summary including:
1. Key findings from each step
2. Most important insights
3. Strategic recommendations
4. Action items

Keep it concise but comprehensive."""),
            ("user", "Complete analysis results: {all_results}")
        ])
    
    def _setup_predefined_workflows(self):
        """Setup common analysis workflows"""
        
        self.workflows = {
            "offensive_efficiency": [
                AnalysisStep("filter_plays", AnalysisStepType.FILTER, 
                           "Filter offensive plays", "Get all offensive plays"),
                AnalysisStep("analyze_formations", AnalysisStepType.AGGREGATE,
                           "Analyze formation efficiency", "Calculate success rates by formation", 
                           depends_on=["filter_plays"]),
                AnalysisStep("compare_situations", AnalysisStepType.COMPARE,
                           "Compare situational performance", "Compare down and distance performance",
                           depends_on=["analyze_formations"]),
                AnalysisStep("identify_trends", AnalysisStepType.TREND,
                           "Identify efficiency trends", "Find patterns in offensive efficiency",
                           depends_on=["compare_situations"]),
                AnalysisStep("generate_insights", AnalysisStepType.INSIGHT,
                           "Generate strategic insights", "Key insights from offensive analysis",
                           depends_on=["identify_trends"]),
                AnalysisStep("recommend_changes", AnalysisStepType.RECOMMENDATION,
                           "Recommend improvements", "Actionable recommendations for offense",
                           depends_on=["generate_insights"])
            ],
            
            "red_zone_analysis": [
                AnalysisStep("filter_red_zone", AnalysisStepType.FILTER,
                           "red zone plays", "Filter plays in red zone (yard line 1-20)"),
                AnalysisStep("analyze_success", AnalysisStepType.AGGREGATE,
                           "Calculate red zone success rates", "Success rate analysis by play type",
                           depends_on=["filter_red_zone"]),
                AnalysisStep("compare_formations", AnalysisStepType.COMPARE,
                           "Compare formation effectiveness", "Compare red zone formations",
                           depends_on=["analyze_success"]),
                AnalysisStep("insights", AnalysisStepType.INSIGHT,
                           "Red zone strategic insights", "Key insights for red zone improvement",
                           depends_on=["compare_formations"]),
                AnalysisStep("recommendations", AnalysisStepType.RECOMMENDATION,
                           "Red zone recommendations", "Specific red zone improvements",
                           depends_on=["insights"])
            ],
            
            "third_down_efficiency": [
                AnalysisStep("filter_third_downs", AnalysisStepType.FILTER,
                           "third down plays", "Filter all third down attempts"),
                AnalysisStep("analyze_by_distance", AnalysisStepType.AGGREGATE,
                           "Analyze by yardage needed", "Success rates by distance categories",
                           depends_on=["filter_third_downs"]),
                AnalysisStep("formation_analysis", AnalysisStepType.COMPARE,
                           "Compare formation success", "Third down formation effectiveness",
                           depends_on=["analyze_by_distance"]),
                AnalysisStep("situational_trends", AnalysisStepType.TREND,
                           "Situational performance trends", "Third down trends by situation",
                           depends_on=["formation_analysis"]),
                AnalysisStep("improvement_insights", AnalysisStepType.INSIGHT,
                           "Areas for improvement", "Key third down insights",
                           depends_on=["situational_trends"]),
                AnalysisStep("tactical_recommendations", AnalysisStepType.RECOMMENDATION,
                           "Tactical recommendations", "Third down strategy improvements",
                           depends_on=["improvement_insights"])
            ]
        }
    
    def execute_workflow(self, workflow_name: str, plays_data: List[Dict]) -> PipelineResult:
        """Execute a predefined analysis workflow"""
        if workflow_name not in self.workflows:
            return PipelineResult(
                pipeline_id=f"unknown_{workflow_name}",
                success=False,
                steps=[],
                summary=f"Unknown workflow: {workflow_name}"
            )
        
        steps = self.workflows[workflow_name]
        return self.execute_custom_pipeline(steps, plays_data, workflow_name)
    
    def execute_custom_pipeline(self, steps: List[AnalysisStep], plays_data: List[Dict], 
                              pipeline_id: str = None) -> PipelineResult:
        """Execute a custom analysis pipeline"""
        if not pipeline_id:
            pipeline_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        start_time = datetime.now()
        step_results = {}
        executed_steps = []
        
        try:
            # Execute steps in dependency order
            for step in self._sort_steps_by_dependencies(steps):
                step_start_time = datetime.now()
                
                # Get previous results for context
                previous_results = self._get_previous_results(step, step_results)
                
                # Execute the step
                step_result = self._execute_step(step, plays_data, previous_results)
                step_result.execution_time = (datetime.now() - step_start_time).total_seconds()
                
                step_results[step.step_id] = step_result
                executed_steps.append(step_result)
                
                # Stop if step failed and is critical
                if not step_result.success:
                    logging.error(f"Step {step.step_id} failed: {step_result.error_message}")
                    break
            
            # Generate pipeline summary
            summary = self._generate_pipeline_summary(executed_steps)
            recommendations = self._extract_recommendations(executed_steps)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                pipeline_id=pipeline_id,
                success=all(step.success for step in executed_steps),
                steps=executed_steps,
                summary=summary,
                recommendations=recommendations,
                total_execution_time=total_time
            )
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            return PipelineResult(
                pipeline_id=pipeline_id,
                success=False,
                steps=executed_steps,
                summary=f"Pipeline failed: {str(e)}",
                total_execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _sort_steps_by_dependencies(self, steps: List[AnalysisStep]) -> List[AnalysisStep]:
        """Sort steps by their dependencies (topological sort)"""
        sorted_steps = []
        remaining_steps = steps.copy()
        step_dict = {step.step_id: step for step in steps}
        
        while remaining_steps:
            # Find steps with no unmet dependencies
            ready_steps = []
            for step in remaining_steps:
                if not step.depends_on:
                    ready_steps.append(step)
                else:
                    # Check if all dependencies are satisfied
                    if all(dep_id in [s.step_id for s in sorted_steps] for dep_id in step.depends_on):
                        ready_steps.append(step)
            
            if not ready_steps:
                # Circular dependency or missing dependency
                logging.warning("Circular dependency detected, executing remaining steps in order")
                sorted_steps.extend(remaining_steps)
                break
            
            # Add ready steps to sorted list
            sorted_steps.extend(ready_steps)
            for step in ready_steps:
                remaining_steps.remove(step)
        
        return sorted_steps
    
    def _execute_step(self, step: AnalysisStep, plays_data: List[Dict], 
                     previous_results: Dict) -> StepResult:
        """Execute a single analysis step"""
        try:
            if step.step_type == AnalysisStepType.FILTER:
                return self._execute_filter_step(step, plays_data, previous_results)
            elif step.step_type == AnalysisStepType.AGGREGATE:
                return self._execute_aggregate_step(step, plays_data, previous_results)
            elif step.step_type == AnalysisStepType.COMPARE:
                return self._execute_comparison_step(step, plays_data, previous_results)
            elif step.step_type == AnalysisStepType.TREND:
                return self._execute_trend_step(step, plays_data, previous_results)
            elif step.step_type == AnalysisStepType.INSIGHT:
                return self._execute_insight_step(step, plays_data, previous_results)
            elif step.step_type == AnalysisStepType.RECOMMENDATION:
                return self._execute_recommendation_step(step, plays_data, previous_results)
            else:
                raise ValueError(f"Unknown step type: {step.step_type}")
                
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                step_type=step.step_type,
                success=False,
                error_message=str(e)
            )
    
    def _execute_filter_step(self, step: AnalysisStep, plays_data: List[Dict], 
                           previous_results: Dict) -> StepResult:
        """Execute a data filtering step"""
        # Translate natural language query to filters
        translation_result = self.query_translator.translate_query(step.query)
        
        if not translation_result.success:
            return StepResult(
                step_id=step.step_id,
                step_type=step.step_type,
                success=False,
                error_message=f"Query translation failed: {translation_result.error_message}"
            )
        
        # Apply filters to data
        filtered_data = self._apply_filters(plays_data, translation_result.filters)
        
        # Generate summary analysis
        data_summary = self._generate_data_summary(filtered_data)
        
        chain = self.filter_template | self.llm | StrOutputParser()
        insights = chain.invoke({
            "step_description": step.description,
            "previous_results": json.dumps(previous_results, indent=2),
            "query": step.query,
            "data_count": len(filtered_data),
            "data_summary": data_summary
        })
        
        return StepResult(
            step_id=step.step_id,
            step_type=step.step_type,
            success=True,
            data={"filtered_data": filtered_data, "count": len(filtered_data)},
            insights=insights,
            metrics={"original_count": len(plays_data), "filtered_count": len(filtered_data)}
        )
    
    def _execute_aggregate_step(self, step: AnalysisStep, plays_data: List[Dict],
                              previous_results: Dict) -> StepResult:
        """Execute aggregation analysis step"""
        # Use filtered data from previous step if available
        working_data = plays_data
        if previous_results and "data" in previous_results:
            if "filtered_data" in previous_results["data"]:
                working_data = previous_results["data"]["filtered_data"]
        
        # Calculate aggregation metrics
        metrics = self._calculate_aggregate_metrics(working_data)
        data_summary = json.dumps(metrics, indent=2)
        
        chain = self.aggregate_template | self.llm | StrOutputParser()
        insights = chain.invoke({
            "step_description": step.description,
            "previous_results": json.dumps(previous_results, indent=2),
            "query": step.query,
            "data_summary": data_summary
        })
        
        return StepResult(
            step_id=step.step_id,
            step_type=step.step_type,
            success=True,
            data={"aggregates": metrics, "data_count": len(working_data)},
            insights=insights,
            metrics=metrics
        )
    
    def _execute_comparison_step(self, step: AnalysisStep, plays_data: List[Dict],
                               previous_results: Dict) -> StepResult:
        """Execute comparison analysis step"""
        # Generate comparison data based on previous results
        comparison_data = self._generate_comparison_groups(plays_data, previous_results, step)
        
        chain = self.comparison_template | self.llm | StrOutputParser()
        insights = chain.invoke({
            "step_description": step.description,
            "previous_results": json.dumps(previous_results, indent=2),
            "query": step.query,
            "comparison_data": json.dumps(comparison_data, indent=2)
        })
        
        return StepResult(
            step_id=step.step_id,
            step_type=step.step_type,
            success=True,
            data={"comparisons": comparison_data},
            insights=insights,
            metrics={"comparison_groups": len(comparison_data)}
        )
    
    def _execute_trend_step(self, step: AnalysisStep, plays_data: List[Dict],
                          previous_results: Dict) -> StepResult:
        """Execute trend analysis step"""
        trend_data = self._analyze_trends(plays_data, previous_results)
        
        chain = self.trend_template | self.llm | StrOutputParser()
        insights = chain.invoke({
            "step_description": step.description,
            "previous_results": json.dumps(previous_results, indent=2),
            "query": step.query,
            "trend_data": json.dumps(trend_data, indent=2)
        })
        
        return StepResult(
            step_id=step.step_id,
            step_type=step.step_type,
            success=True,
            data={"trends": trend_data},
            insights=insights,
            metrics={"trend_indicators": len(trend_data)}
        )
    
    def _execute_insight_step(self, step: AnalysisStep, plays_data: List[Dict],
                            previous_results: Dict) -> StepResult:
        """Execute insight generation step"""
        analysis_data = self._compile_analysis_data(previous_results)
        
        chain = self.insight_template | self.llm | StrOutputParser()
        insights = chain.invoke({
            "step_description": step.description,
            "previous_results": json.dumps(previous_results, indent=2),
            "query": step.query,
            "analysis_data": json.dumps(analysis_data, indent=2)
        })
        
        return StepResult(
            step_id=step.step_id,
            step_type=step.step_type,
            success=True,
            data={"compiled_analysis": analysis_data},
            insights=insights
        )
    
    def _execute_recommendation_step(self, step: AnalysisStep, plays_data: List[Dict],
                                   previous_results: Dict) -> StepResult:
        """Execute recommendation generation step"""
        final_data = self._compile_final_analysis(previous_results)
        
        chain = self.recommendation_template | self.llm | StrOutputParser()
        recommendations = chain.invoke({
            "step_description": step.description,
            "previous_results": json.dumps(previous_results, indent=2),
            "query": step.query,
            "final_data": json.dumps(final_data, indent=2)
        })
        
        return StepResult(
            step_id=step.step_id,
            step_type=step.step_type,
            success=True,
            data={"final_analysis": final_data},
            insights=recommendations
        )
    
    # Helper methods for data processing and analysis
    def _apply_filters(self, plays_data: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to plays data"""
        if not filters or not filters.get("conditions"):
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
                
                if operator == "equals" and play_value != value:
                    match = False
                elif operator == "greater_than" and play_value <= value:
                    match = False
                elif operator == "less_than" and play_value >= value:
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
        """Generate summary of plays data"""
        if not plays_data:
            return "No data available"
        
        total_plays = len(plays_data)
        total_yards = sum(play.get('yards_gained', 0) for play in plays_data)
        avg_yards = total_yards / total_plays if total_plays > 0 else 0
        
        play_types = {}
        formations = {}
        downs = {}
        
        for play in plays_data:
            play_type = play.get('play_type', 'Unknown')
            formation = play.get('formation', 'Unknown')
            down = play.get('down', 0)
            
            play_types[play_type] = play_types.get(play_type, 0) + 1
            formations[formation] = formations.get(formation, 0) + 1
            if down:
                downs[f"Down {down}"] = downs.get(f"Down {down}", 0) + 1
        
        return f"""Total Plays: {total_plays}
Total Yards: {total_yards}
Average Yards/Play: {avg_yards:.2f}
Top Play Types: {dict(sorted(play_types.items(), key=lambda x: x[1], reverse=True)[:3])}
Top Formations: {dict(sorted(formations.items(), key=lambda x: x[1], reverse=True)[:3])}
Down Distribution: {downs}"""
    
    def _calculate_aggregate_metrics(self, plays_data: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics for plays data"""
        if not plays_data:
            return {}
        
        metrics = {
            "total_plays": len(plays_data),
            "total_yards": sum(play.get('yards_gained', 0) for play in plays_data),
            "average_yards_per_play": 0,
            "success_rate": 0,
            "by_play_type": {},
            "by_formation": {},
            "by_down": {}
        }
        
        if metrics["total_plays"] > 0:
            metrics["average_yards_per_play"] = metrics["total_yards"] / metrics["total_plays"]
            successful_plays = len([p for p in plays_data if p.get('yards_gained', 0) > 0])
            metrics["success_rate"] = successful_plays / metrics["total_plays"]
        
        # Aggregate by categories
        for play in plays_data:
            play_type = play.get('play_type', 'Unknown')
            formation = play.get('formation', 'Unknown')
            down = play.get('down', 0)
            yards = play.get('yards_gained', 0)
            
            # By play type
            if play_type not in metrics["by_play_type"]:
                metrics["by_play_type"][play_type] = {"count": 0, "total_yards": 0, "avg_yards": 0}
            metrics["by_play_type"][play_type]["count"] += 1
            metrics["by_play_type"][play_type]["total_yards"] += yards
            metrics["by_play_type"][play_type]["avg_yards"] = (
                metrics["by_play_type"][play_type]["total_yards"] / 
                metrics["by_play_type"][play_type]["count"]
            )
            
            # Similar for formation and down...
            
        return metrics
    
    def _generate_comparison_groups(self, plays_data: List[Dict], previous_results: Dict, 
                                  step: AnalysisStep) -> Dict[str, Any]:
        """Generate comparison groups based on context"""
        # This is a simplified implementation
        # In practice, this would be more sophisticated based on step context
        
        comparisons = {}
        
        # Compare by play type
        play_types = {}
        for play in plays_data:
            play_type = play.get('play_type', 'Unknown')
            if play_type not in play_types:
                play_types[play_type] = []
            play_types[play_type].append(play)
        
        comparisons["by_play_type"] = {}
        for play_type, plays in play_types.items():
            if len(plays) > 0:
                avg_yards = sum(p.get('yards_gained', 0) for p in plays) / len(plays)
                success_rate = len([p for p in plays if p.get('yards_gained', 0) > 0]) / len(plays)
                comparisons["by_play_type"][play_type] = {
                    "count": len(plays),
                    "avg_yards": avg_yards,
                    "success_rate": success_rate
                }
        
        return comparisons
    
    def _analyze_trends(self, plays_data: List[Dict], previous_results: Dict) -> Dict[str, Any]:
        """Analyze trends in the data"""
        # Simplified trend analysis
        trends = {
            "performance_by_quarter": {},
            "efficiency_trends": {},
            "situational_patterns": {}
        }
        
        # This would be more sophisticated in practice
        return trends
    
    def _compile_analysis_data(self, previous_results: Dict) -> Dict[str, Any]:
        """Compile analysis data from previous steps"""
        compiled = {
            "key_metrics": {},
            "important_findings": [],
            "data_insights": {}
        }
        
        # Extract key information from previous steps
        for step_id, result in previous_results.items():
            if hasattr(result, 'metrics') and result.metrics:
                compiled["key_metrics"][step_id] = result.metrics
            if hasattr(result, 'insights') and result.insights:
                compiled["important_findings"].append({
                    "step": step_id,
                    "insight": result.insights[:200] + "..." if len(result.insights) > 200 else result.insights
                })
        
        return compiled
    
    def _compile_final_analysis(self, previous_results: Dict) -> Dict[str, Any]:
        """Compile final analysis for recommendations"""
        return self._compile_analysis_data(previous_results)
    
    def _get_previous_results(self, step: AnalysisStep, step_results: Dict) -> Dict:
        """Get previous results for context"""
        previous = {}
        if step.depends_on:
            for dep_id in step.depends_on:
                if dep_id in step_results:
                    previous[dep_id] = asdict(step_results[dep_id])
        return previous
    
    def _generate_pipeline_summary(self, steps: List[StepResult]) -> str:
        """Generate overall pipeline summary"""
        all_results = [asdict(step) for step in steps]
        
        chain = self.summary_template | self.llm | StrOutputParser()
        summary = chain.invoke({
            "all_results": json.dumps(all_results, indent=2)
        })
        
        return summary
    
    def _extract_recommendations(self, steps: List[StepResult]) -> List[str]:
        """Extract recommendations from pipeline steps"""
        recommendations = []
        for step in steps:
            if step.step_type == AnalysisStepType.RECOMMENDATION and step.insights:
                # Extract bullet points or numbered items from insights
                lines = step.insights.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith(('•', '-', '*')) or line[0:2].isdigit():
                        recommendations.append(line)
        return recommendations
    
    def get_available_workflows(self) -> Dict[str, List[str]]:
        """Get available predefined workflows"""
        workflows = {}
        for name, steps in self.workflows.items():
            workflows[name] = [f"{step.step_id}: {step.description}" for step in steps]
        return workflows