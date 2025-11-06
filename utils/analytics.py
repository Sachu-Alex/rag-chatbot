"""
Analytics and monitoring utilities for the Document Q&A System.
"""
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter

# Local imports
from config import config

logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

class QAAnalytics:
    """Analytics manager for Q&A system performance and usage."""
    
    def __init__(self, analytics_dir: str = "./analytics"):
        """Initialize analytics manager."""
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(exist_ok=True)
        
        # Analytics data files
        self.query_log_file = self.analytics_dir / "query_log.jsonl"
        self.performance_log_file = self.analytics_dir / "performance_log.jsonl"
        self.usage_stats_file = self.analytics_dir / "usage_stats.json"
        
        # In-memory storage for current session
        self.current_session = {
            "queries": [],
            "performance_metrics": [],
            "start_time": datetime.now(),
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    def log_query(self, 
                  question: str, 
                  answer: str, 
                  sources: List[Dict[str, Any]], 
                  retrieval_strategy: str,
                  response_time: float,
                  conversation_used: bool = False,
                  metadata: Optional[Dict[str, Any]] = None):
        """Log a Q&A interaction."""
        try:
            query_data = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.current_session["session_id"],
                "question": question,
                "question_length": len(question),
                "answer": answer,
                "answer_length": len(answer),
                "sources_count": len(sources),
                "retrieval_strategy": retrieval_strategy,
                "response_time": response_time,
                "conversation_used": conversation_used,
                "metadata": metadata or {}
            }
            
            # Add to current session
            self.current_session["queries"].append(query_data)
            
            # Append to log file
            with open(self.query_log_file, "a") as f:
                f.write(json.dumps(query_data) + "\n")
            
            logger.debug(f"Query logged: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
    
    def log_performance(self, 
                       operation: str, 
                       duration: float, 
                       success: bool,
                       metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        try:
            performance_data = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.current_session["session_id"],
                "operation": operation,
                "duration": duration,
                "success": success,
                "metadata": metadata or {}
            }
            
            # Add to current session
            self.current_session["performance_metrics"].append(performance_data)
            
            # Append to log file
            with open(self.performance_log_file, "a") as f:
                f.write(json.dumps(performance_data) + "\n")
            
            logger.debug(f"Performance logged: {operation} - {duration:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to log performance: {e}")
    
    def get_query_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get query statistics for the specified number of days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            queries = self._load_queries_since(cutoff_date)
            
            if not queries:
                return {"total_queries": 0, "period_days": days}
            
            # Basic statistics
            total_queries = len(queries)
            avg_question_length = sum(q.get("question_length", 0) for q in queries) / total_queries
            avg_answer_length = sum(q.get("answer_length", 0) for q in queries) / total_queries
            avg_response_time = sum(q.get("response_time", 0) for q in queries) / total_queries
            avg_sources_count = sum(q.get("sources_count", 0) for q in queries) / total_queries
            
            # Strategy usage
            strategy_counts = Counter(q.get("retrieval_strategy", "unknown") for q in queries)
            
            # Conversation usage
            conversation_queries = sum(1 for q in queries if q.get("conversation_used", False))
            
            # Time-based analysis
            queries_by_hour = defaultdict(int)
            for query in queries:
                try:
                    hour = datetime.fromisoformat(query["timestamp"]).hour
                    queries_by_hour[hour] += 1
                except:
                    continue
            
            # Response time percentiles
            response_times = [q.get("response_time", 0) for q in queries if q.get("response_time")]
            if response_times:
                response_times.sort()
                p50 = response_times[len(response_times) // 2]
                p95 = response_times[int(len(response_times) * 0.95)]
                p99 = response_times[int(len(response_times) * 0.99)]
            else:
                p50 = p95 = p99 = 0
            
            return {
                "period_days": days,
                "total_queries": total_queries,
                "avg_question_length": round(avg_question_length, 2),
                "avg_answer_length": round(avg_answer_length, 2),
                "avg_response_time": round(avg_response_time, 3),
                "avg_sources_count": round(avg_sources_count, 2),
                "strategy_usage": dict(strategy_counts),
                "conversation_queries": conversation_queries,
                "conversation_percentage": round(conversation_queries / total_queries * 100, 2) if total_queries > 0 else 0,
                "queries_by_hour": dict(queries_by_hour),
                "response_time_percentiles": {
                    "p50": round(p50, 3),
                    "p95": round(p95, 3),
                    "p99": round(p99, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get query statistics: {e}")
            return {"error": str(e)}
    
    def get_performance_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance statistics for the specified number of days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            metrics = self._load_performance_since(cutoff_date)
            
            if not metrics:
                return {"total_operations": 0, "period_days": days}
            
            # Basic statistics
            total_operations = len(metrics)
            successful_operations = sum(1 for m in metrics if m.get("success", False))
            success_rate = successful_operations / total_operations * 100 if total_operations > 0 else 0
            
            # Operation type analysis
            operation_counts = Counter(m.get("operation", "unknown") for m in metrics)
            
            # Performance by operation
            operation_performance = defaultdict(list)
            for metric in metrics:
                operation = metric.get("operation", "unknown")
                duration = metric.get("duration", 0)
                if duration > 0:
                    operation_performance[operation].append(duration)
            
            # Calculate averages and percentiles for each operation
            operation_stats = {}
            for operation, durations in operation_performance.items():
                if durations:
                    durations.sort()
                    operation_stats[operation] = {
                        "count": len(durations),
                        "avg_duration": round(sum(durations) / len(durations), 3),
                        "p50": round(durations[len(durations) // 2], 3),
                        "p95": round(durations[int(len(durations) * 0.95)], 3)
                    }
            
            return {
                "period_days": days,
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "success_rate": round(success_rate, 2),
                "operation_counts": dict(operation_counts),
                "operation_statistics": operation_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance statistics: {e}")
            return {"error": str(e)}
    
    def get_usage_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get usage trends over time."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            queries = self._load_queries_since(cutoff_date)
            
            if not queries:
                return {"period_days": days, "daily_usage": {}}
            
            # Daily usage
            daily_usage = defaultdict(int)
            for query in queries:
                try:
                    date = datetime.fromisoformat(query["timestamp"]).date()
                    daily_usage[date.isoformat()] += 1
                except:
                    continue
            
            # Weekly patterns
            weekly_patterns = defaultdict(int)
            for query in queries:
                try:
                    weekday = datetime.fromisoformat(query["timestamp"]).strftime("%A")
                    weekly_patterns[weekday] += 1
                except:
                    continue
            
            # Hourly patterns
            hourly_patterns = defaultdict(int)
            for query in queries:
                try:
                    hour = datetime.fromisoformat(query["timestamp"]).hour
                    hourly_patterns[hour] += 1
                except:
                    continue
            
            return {
                "period_days": days,
                "total_queries": len(queries),
                "daily_usage": dict(daily_usage),
                "weekly_patterns": dict(weekly_patterns),
                "hourly_patterns": dict(hourly_patterns)
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage trends: {e}")
            return {"error": str(e)}
    
    def get_question_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Analyze question patterns and characteristics."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            queries = self._load_queries_since(cutoff_date)
            
            if not queries:
                return {"total_questions": 0, "period_days": days}
            
            questions = [q.get("question", "") for q in queries]
            
            # Length analysis
            lengths = [len(q) for q in questions]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            
            # Word frequency analysis
            all_words = []
            for question in questions:
                words = question.lower().split()
                all_words.extend([word.strip(".,!?;:") for word in words if len(word) > 2])
            
            word_frequency = Counter(all_words)
            most_common_words = word_frequency.most_common(20)
            
            # Question type patterns
            question_starters = []
            for question in questions:
                if question:
                    first_word = question.split()[0].lower()
                    question_starters.append(first_word)
            
            starter_frequency = Counter(question_starters)
            
            # Question word analysis
            question_words = ["what", "how", "why", "when", "where", "who", "which"]
            question_word_counts = {}
            
            for word in question_words:
                count = sum(1 for q in questions if word.lower() in q.lower())
                if count > 0:
                    question_word_counts[word] = count
            
            return {
                "period_days": days,
                "total_questions": len(questions),
                "avg_question_length": round(avg_length, 2),
                "length_distribution": {
                    "min": min(lengths) if lengths else 0,
                    "max": max(lengths) if lengths else 0,
                    "median": sorted(lengths)[len(lengths)//2] if lengths else 0
                },
                "most_common_words": most_common_words,
                "question_starters": dict(starter_frequency.most_common(10)),
                "question_word_usage": question_word_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze questions: {e}")
            return {"error": str(e)}
    
    def _load_queries_since(self, cutoff_date: datetime) -> List[Dict[str, Any]]:
        """Load queries since the specified date."""
        queries = []
        
        if self.query_log_file.exists():
            try:
                with open(self.query_log_file, "r") as f:
                    for line in f:
                        try:
                            query = json.loads(line.strip())
                            query_date = datetime.fromisoformat(query["timestamp"])
                            if query_date >= cutoff_date:
                                queries.append(query)
                        except:
                            continue
            except Exception as e:
                logger.error(f"Failed to load queries: {e}")
        
        # Add current session queries
        for query in self.current_session["queries"]:
            try:
                query_date = datetime.fromisoformat(query["timestamp"])
                if query_date >= cutoff_date:
                    queries.append(query)
            except:
                continue
        
        return queries
    
    def _load_performance_since(self, cutoff_date: datetime) -> List[Dict[str, Any]]:
        """Load performance metrics since the specified date."""
        metrics = []
        
        if self.performance_log_file.exists():
            try:
                with open(self.performance_log_file, "r") as f:
                    for line in f:
                        try:
                            metric = json.loads(line.strip())
                            metric_date = datetime.fromisoformat(metric["timestamp"])
                            if metric_date >= cutoff_date:
                                metrics.append(metric)
                        except:
                            continue
            except Exception as e:
                logger.error(f"Failed to load performance metrics: {e}")
        
        # Add current session metrics
        for metric in self.current_session["performance_metrics"]:
            try:
                metric_date = datetime.fromisoformat(metric["timestamp"])
                if metric_date >= cutoff_date:
                    metrics.append(metric)
            except:
                continue
        
        return metrics
    
    def generate_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a comprehensive analytics report."""
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "period_days": days,
                "query_statistics": self.get_query_statistics(days),
                "performance_statistics": self.get_performance_statistics(days),
                "usage_trends": self.get_usage_trends(days),
                "question_analysis": self.get_question_analysis(days)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {"error": str(e)}
    
    def export_report(self, days: int = 7, output_file: Optional[str] = None) -> str:
        """Export analytics report to JSON file."""
        try:
            report = self.generate_report(days)
            
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.analytics_dir / f"analytics_report_{timestamp}.json"
            
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Analytics report exported to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return ""
    
    def clear_old_logs(self, days_to_keep: int = 30):
        """Clear log entries older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean query logs
            if self.query_log_file.exists():
                temp_file = self.query_log_file.with_suffix(".tmp")
                
                with open(self.query_log_file, "r") as infile, open(temp_file, "w") as outfile:
                    for line in infile:
                        try:
                            query = json.loads(line.strip())
                            query_date = datetime.fromisoformat(query["timestamp"])
                            if query_date >= cutoff_date:
                                outfile.write(line)
                        except:
                            continue
                
                temp_file.replace(self.query_log_file)
            
            # Clean performance logs
            if self.performance_log_file.exists():
                temp_file = self.performance_log_file.with_suffix(".tmp")
                
                with open(self.performance_log_file, "r") as infile, open(temp_file, "w") as outfile:
                    for line in infile:
                        try:
                            metric = json.loads(line.strip())
                            metric_date = datetime.fromisoformat(metric["timestamp"])
                            if metric_date >= cutoff_date:
                                outfile.write(line)
                        except:
                            continue
                
                temp_file.replace(self.performance_log_file)
            
            logger.info(f"Cleared logs older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Failed to clear old logs: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        session_duration = (datetime.now() - self.current_session["start_time"]).total_seconds()
        
        return {
            "session_id": self.current_session["session_id"],
            "start_time": self.current_session["start_time"].isoformat(),
            "duration_seconds": round(session_duration, 2),
            "queries_in_session": len(self.current_session["queries"]),
            "performance_logs_in_session": len(self.current_session["performance_metrics"])
        }