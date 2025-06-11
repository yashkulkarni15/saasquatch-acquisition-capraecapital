"""
SaaSQuatch Acquisition Intelligence Engine - Enhanced Version
AI-Powered M&A Deal Sourcing and Scoring Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import json
import random
from dataclasses import dataclass
from enum import Enum

class DealTiming(Enum):
    BUY = "BUY_NOW"
    WAIT = "WAIT"
    PASS = "PASS"

@dataclass
class DealStructure:
    suggested_multiple: float
    earnout_percentage: float
    transition_months: int
    confidence_score: float
    rationale: str

class AcquisitionScorer:
    """
    AI-Enhanced scoring system for acquisition targets
    Combines traditional metrics with ML predictions
    """
    
    def __init__(self):
        # Enhanced weights with AI adjustment capability
        self.base_weights = {
            'owner_readiness': 0.30,
            'financial_health': 0.25,
            'valuation_reason': 0.20,
            'business_quality': 0.15,
            'transition_ease': 0.10
        }
        self.weights = self.base_weights.copy()
        
        # Load ML model and industry benchmarks
        self.ml_model = self._initialize_ml_model()
        self.industry_benchmarks = self._load_industry_benchmarks()
        self.success_patterns = self._load_success_patterns()
    
    def _initialize_ml_model(self):
        """Initialize a simple ML model for pattern recognition"""
        # In production, this would load a trained model
        return {
            'success_threshold': 0.75,
            'feature_importance': {
                'owner_age': 0.15,
                'ebitda_margin': 0.20,
                'revenue_growth': 0.15,
                'asking_multiple': 0.20,
                'market_timing': 0.30
            }
        }
    
    def _load_industry_benchmarks(self):
        """Load industry-specific benchmarks"""
        return {
            'SaaS': {'typical_multiple': 5.5, 'growth_rate': 30, 'margin': 25},
            'E-commerce': {'typical_multiple': 3.5, 'growth_rate': 20, 'margin': 15},
            'Manufacturing': {'typical_multiple': 4.0, 'growth_rate': 10, 'margin': 18},
            'Healthcare': {'typical_multiple': 6.0, 'growth_rate': 15, 'margin': 20},
            'Professional Services': {'typical_multiple': 4.5, 'growth_rate': 15, 'margin': 22}
        }
    
    def _load_success_patterns(self):
        """Historical patterns from successful acquisitions"""
        return {
            'optimal_owner_age_range': (55, 65),
            'optimal_ownership_duration': (8, 15),
            'optimal_employee_count': (50, 200),
            'warning_signals': ['declining_growth', 'customer_concentration', 'owner_dependent']
        }
    
    def score_target_with_ai(self, company: Dict) -> Tuple[int, List[Dict], Dict]:
        """
        Enhanced scoring with AI predictions and insights
        Returns: (score, signals, ai_insights)
        """
        # Traditional scoring
        base_score, signals = self.score_target(company)
        
        # AI enhancements
        ml_prediction = self._get_ml_prediction(company)
        market_timing = self._analyze_market_timing(company)
        peer_comparison = self._benchmark_against_peers(company)
        
        # Adjust score based on AI insights
        ai_adjusted_score = self._adjust_score_with_ai(
            base_score, ml_prediction, market_timing
        )
        
        ai_insights = {
            'ml_success_probability': ml_prediction,
            'market_timing': market_timing,
            'peer_percentile': peer_comparison['percentile'],
            'hidden_opportunities': self._find_hidden_patterns(company),
            'risk_factors': self._identify_ai_risks(company)
        }
        
        return int(ai_adjusted_score), signals, ai_insights
    
    def _get_ml_prediction(self, company: Dict) -> float:
        """Predict acquisition success using ML model"""
        features = {
            'owner_age_score': self._score_age_ml(company.get('owner_age', 0)),
            'financial_score': self._score_financials_ml(company),
            'market_position': self._score_market_position_ml(company),
            'timing_score': self._score_timing_ml(company)
        }
        
        # Simple weighted prediction (in production, use real ML)
        prediction = sum(
            features[key] * self.ml_model['feature_importance'].get(key.split('_')[0], 0.1)
            for key in features
        )
        
        return min(0.95, max(0.05, prediction))
    
    def _analyze_market_timing(self, company: Dict) -> Dict:
        """Analyze if it's the right time to acquire"""
        industry = company.get('industry', 'General')
        
        # Market timing factors
        factors = {
            'industry_consolidation': self._check_consolidation_trend(industry),
            'valuation_cycle': self._check_valuation_cycle(company),
            'competitive_landscape': self._analyze_competition(company),
            'macro_factors': self._check_macro_environment()
        }
        
        # Calculate timing score
        timing_score = sum(factors.values()) / len(factors)
        
        if timing_score > 0.7:
            recommendation = DealTiming.BUY
        elif timing_score > 0.4:
            recommendation = DealTiming.WAIT
        else:
            recommendation = DealTiming.PASS
            
        return {
            'recommendation': recommendation.value,
            'confidence': timing_score,
            'factors': factors,
            'optimal_window': self._calculate_optimal_window(factors)
        }
    
    def score_target(self, company: Dict) -> Tuple[int, List[Dict]]:
        """Original scoring method - enhanced with better logic"""
        signals = []
        component_scores = {}
        
        # Enhanced scoring for each component
        for component in ['owner_readiness', 'financial_health', 'valuation_reason', 
                         'business_quality', 'transition_ease']:
            score_method = getattr(self, f'_score_{component}')
            score, component_signals = score_method(company)
            component_scores[component] = score
            signals.extend(component_signals)
        
        # Add confidence indicator to each signal
        for signal in signals:
            signal['confidence'] = self._calculate_signal_confidence(signal, company)
        
        # Calculate weighted total
        total_score = sum(
            component_scores[key] * self.weights[key] * 100
            for key in self.weights
        )
        
        return int(total_score), signals
    
    def _calculate_signal_confidence(self, signal: Dict, company: Dict) -> float:
        """Calculate confidence level for each signal"""
        # Based on data completeness and historical accuracy
        base_confidence = 0.7
        
        if signal['type'] == 'positive':
            base_confidence += 0.1
        elif signal['type'] == 'warning':
            base_confidence -= 0.1
            
        # Adjust based on data quality
        data_quality = self._assess_data_quality(company)
        return min(0.95, base_confidence * data_quality)
    
    def _benchmark_against_peers(self, company: Dict) -> Dict:
        """Compare against successful acquisitions in same industry"""
        industry = company.get('industry', 'General')
        benchmarks = self.industry_benchmarks.get(industry, self.industry_benchmarks['SaaS'])
        
        # Calculate percentile rankings
        metrics = {
            'ebitda_margin': (company.get('ebitda', 0) / max(company.get('revenue', 1), 1)) * 100,
            'growth_rate': company.get('revenue_growth_rate', 0),
            'valuation_multiple': company.get('asking_price', 0) / max(company.get('ebitda', 1), 1)
        }
        
        percentiles = {}
        for metric, value in metrics.items():
            benchmark = benchmarks.get(metric.split('_')[0], value)
            percentiles[metric] = min(100, (value / benchmark) * 50) if benchmark > 0 else 50
            
        return {
            'percentile': np.mean(list(percentiles.values())),
            'strengths': [k for k, v in percentiles.items() if v > 70],
            'weaknesses': [k for k, v in percentiles.items() if v < 30],
            'metrics': metrics
        }
    
    # Enhanced component scoring methods
    def _score_owner_readiness(self, company: Dict) -> Tuple[float, List[Dict]]:
        """Enhanced owner readiness evaluation with AI insights"""
        score = 0
        signals = []
        
        owner_age = company.get('owner_age', 0)
        years_owned = company.get('years_owned', 0)
        
        # Age-based scoring with nuance
        if owner_age >= 60:
            score += 0.5
            urgency = "high" if owner_age >= 65 else "medium"
            signals.append({
                'type': 'positive',
                'text': f'Owner approaching retirement age ({owner_age}) - {urgency} urgency',
                'impact': 'high'
            })
        elif owner_age >= 55:
            score += 0.3
            signals.append({
                'type': 'neutral',
                'text': f'Owner age {owner_age} - potential exit in 3-5 years',
                'impact': 'medium'
            })
        elif owner_age >= 50:
            score += 0.1
            signals.append({
                'type': 'neutral',
                'text': f'Owner age {owner_age} - early-stage exit planning possible',
                'impact': 'low'
            })
        else:
            signals.append({
                'type': 'warning',
                'text': f'Young owner ({owner_age}) - likely focused on growth, not exit',
                'impact': 'high'
            })
        
        # Ownership duration insights
        if years_owned > 15:
            score += 0.3
            signals.append({
                'type': 'positive',
                'text': f'Very long tenure ({years_owned} years) - potential fatigue',
                'impact': 'high'
            })
        elif years_owned > 10:
            score += 0.2
            signals.append({
                'type': 'positive',
                'text': f'Long ownership tenure ({years_owned} years) - may consider exit',
                'impact': 'medium'
            })
        elif years_owned > 7:
            score += 0.1
            signals.append({
                'type': 'neutral',
                'text': f'Moderate tenure ({years_owned} years) - timing depends on other factors',
                'impact': 'low'
            })
            
        # Additional readiness indicators
        if company.get('actively_selling', False):
            score += 0.3
            signals.append({
                'type': 'positive',
                'text': 'Business actively listed for sale - immediate opportunity',
                'impact': 'high'
            })
            
        if company.get('hired_investment_banker', False):
            score += 0.2
            signals.append({
                'type': 'positive',
                'text': 'Investment banker engaged - serious about exit',
                'impact': 'high'
            })
            
        if company.get('family_business', False) and owner_age > 55:
            score += 0.1
            signals.append({
                'type': 'positive',
                'text': 'Family business with aging owner - succession concerns likely',
                'impact': 'medium'
            })
            
        return min(1.0, score), signals
    
    def _score_financial_health(self, company: Dict) -> Tuple[float, List[Dict]]:
        """Enhanced financial evaluation with trend analysis"""
        score = 0
        signals = []
        
        revenue = company.get('revenue', 0)
        ebitda = company.get('ebitda', 0)
        recurring_revenue_pct = company.get('recurring_revenue_pct', 0)
        
        if revenue > 0:
            margin = (ebitda / revenue) * 100
            
            # Industry-adjusted margin scoring
            industry = company.get('industry', 'General')
            industry_benchmark = self.industry_benchmarks.get(
                industry, {'margin': 20}
            )['margin']
            
            margin_vs_industry = margin / industry_benchmark if industry_benchmark > 0 else 1
            
            if margin_vs_industry >= 1.3:
                score += 0.4
                signals.append({
                    'type': 'positive',
                    'text': f'{margin:.1f}% EBITDA margin - {(margin_vs_industry-1)*100:.0f}% above industry average',
                    'impact': 'high'
                })
            elif margin_vs_industry >= 1.1:
                score += 0.3
                signals.append({
                    'type': 'positive',
                    'text': f'{margin:.1f}% EBITDA margin - above industry average',
                    'impact': 'medium'
                })
            elif margin_vs_industry >= 0.9:
                score += 0.2
                signals.append({
                    'type': 'neutral',
                    'text': f'{margin:.1f}% EBITDA margin - near industry average',
                    'impact': 'low'
                })
            else:
                signals.append({
                    'type': 'warning',
                    'text': f'{margin:.1f}% EBITDA margin - {(1-margin_vs_industry)*100:.0f}% below industry average',
                    'impact': 'high'
                })
        
        # Enhanced recurring revenue analysis
        if recurring_revenue_pct > 80:
            score += 0.35
            signals.append({
                'type': 'positive',
                'text': f'{recurring_revenue_pct}% recurring revenue - highly predictable',
                'impact': 'high'
            })
        elif recurring_revenue_pct > 60:
            score += 0.25
            signals.append({
                'type': 'positive',
                'text': f'{recurring_revenue_pct}% recurring revenue - good stability',
                'impact': 'medium'
            })
        elif recurring_revenue_pct > 40:
            score += 0.15
            signals.append({
                'type': 'neutral',
                'text': f'{recurring_revenue_pct}% recurring revenue - moderate predictability',
                'impact': 'medium'
            })
        
        # Growth trajectory analysis
        growth_rate = company.get('revenue_growth_rate', 0)
        growth_trend = company.get('growth_trend', 'stable')
        
        if growth_rate > 30 and growth_trend == 'accelerating':
            score += 0.35
            signals.append({
                'type': 'positive',
                'text': f'Accelerating growth at {growth_rate}% YoY - strong momentum',
                'impact': 'high'
            })
        elif growth_rate > 20:
            score += 0.25
            signals.append({
                'type': 'positive',
                'text': f'Consistent {growth_rate}% YoY growth',
                'impact': 'medium'
            })
        elif growth_rate > 10:
            score += 0.15
            signals.append({
                'type': 'neutral',
                'text': f'Moderate growth at {growth_rate}% YoY',
                'impact': 'low'
            })
        elif growth_rate < 5 and growth_trend == 'declining':
            signals.append({
                'type': 'warning',
                'text': f'Declining growth trajectory - potential stagnation',
                'impact': 'high'
            })
            
        return min(1.0, score), signals
    
    # AI-powered helper methods
    def _score_age_ml(self, age: int) -> float:
        """ML-based age scoring"""
        optimal_range = self.success_patterns['optimal_owner_age_range']
        if optimal_range[0] <= age <= optimal_range[1]:
            return 0.9
        elif age >= optimal_range[1]:
            return 0.7
        elif age >= optimal_range[0] - 5:
            return 0.5
        return 0.2
    
    def _score_financials_ml(self, company: Dict) -> float:
        """ML-based financial scoring"""
        revenue = company.get('revenue', 0)
        ebitda = company.get('ebitda', 0)
        growth = company.get('revenue_growth_rate', 0)
        
        if revenue == 0:
            return 0.1
            
        margin = (ebitda / revenue) * 100
        
        # Composite financial score
        margin_score = min(1.0, margin / 30)
        growth_score = min(1.0, growth / 30)
        size_score = min(1.0, revenue / 50000000)
        
        return (margin_score * 0.4 + growth_score * 0.4 + size_score * 0.2)
    
    def _check_consolidation_trend(self, industry: str) -> float:
        """Check if industry is consolidating"""
        # In production, this would use real market data
        consolidation_rates = {
            'SaaS': 0.8,
            'E-commerce': 0.6,
            'Manufacturing': 0.5,
            'Healthcare': 0.7,
            'Professional Services': 0.6
        }
        return consolidation_rates.get(industry, 0.5)
    
    def _find_hidden_patterns(self, company: Dict) -> List[str]:
        """Identify non-obvious opportunities using AI"""
        patterns = []
        
        # Check for hidden value indicators
        if company.get('r_and_d_spend', 0) > company.get('revenue', 1) * 0.15:
            patterns.append("High R&D spend indicates potential IP value")
            
        if company.get('customer_retention', 0) > 95:
            patterns.append("Exceptional customer retention suggests pricing power")
            
        if company.get('employee_tenure_avg', 0) > 5:
            patterns.append("Long employee tenure indicates stable operations")
            
        return patterns
    
    def _identify_ai_risks(self, company: Dict) -> List[Dict]:
        """AI-identified risk factors"""
        risks = []
        
        # Technology risk
        if company.get('tech_stack_age', 0) > 5:
            risks.append({
                'type': 'technology',
                'severity': 'medium',
                'description': 'Aging technology stack may require investment'
            })
            
        # Market risk
        if company.get('market_share_trend') == 'declining':
            risks.append({
                'type': 'market',
                'severity': 'high',
                'description': 'Declining market share indicates competitive pressure'
            })
            
        return risks
    
    # Additional helper methods
    def _check_valuation_cycle(self, company: Dict) -> float:
        """Check where valuations are in the cycle"""
        # Simplified - in production would use real market data
        return random.uniform(0.4, 0.8)
    
    def _analyze_competition(self, company: Dict) -> float:
        """Analyze competitive landscape"""
        return random.uniform(0.5, 0.9)
    
    def _check_macro_environment(self) -> float:
        """Check macroeconomic factors"""
        return random.uniform(0.4, 0.7)
    
    def _calculate_optimal_window(self, factors: Dict) -> str:
        """Calculate optimal acquisition window"""
        avg_score = sum(factors.values()) / len(factors)
        if avg_score > 0.7:
            return "0-3 months"
        elif avg_score > 0.5:
            return "3-6 months"
        return "6-12 months"
    
    def _adjust_score_with_ai(self, base_score: int, ml_prediction: float, 
                             market_timing: Dict) -> float:
        """Adjust score based on AI insights"""
        # Weight the different components
        timing_multiplier = 1.0
        if market_timing['recommendation'] == DealTiming.BUY.value:
            timing_multiplier = 1.15
        elif market_timing['recommendation'] == DealTiming.WAIT.value:
            timing_multiplier = 0.95
        else:
            timing_multiplier = 0.8
            
        # Combine base score with ML prediction
        adjusted = (base_score * 0.7 + ml_prediction * 100 * 0.3) * timing_multiplier
        
        return min(100, max(0, adjusted))
    
    def _assess_data_quality(self, company: Dict) -> float:
        """Assess the quality/completeness of data"""
        required_fields = ['revenue', 'ebitda', 'owner_age', 'employees', 'year_founded']
        optional_fields = ['recurring_revenue_pct', 'growth_trend', 'customer_concentration']
        
        required_complete = sum(1 for field in required_fields if company.get(field)) / len(required_fields)
        optional_complete = sum(1 for field in optional_fields if company.get(field)) / len(optional_fields)
        
        return required_complete * 0.7 + optional_complete * 0.3
    
    # Existing methods continue with enhancements...
    def _score_valuation(self, company: Dict) -> Tuple[float, List[Dict]]:
        """Enhanced valuation scoring with market context"""
        score = 0
        signals = []
        
        ebitda = company.get('ebitda', 1)
        asking_price = company.get('asking_price', 0)
        industry = company.get('industry', 'General')
        
        if asking_price > 0 and ebitda > 0:
            multiple = asking_price / ebitda
            industry_avg = self.industry_benchmarks.get(
                industry, {'typical_multiple': 5}
            )['typical_multiple']
            
            relative_multiple = multiple / industry_avg if industry_avg > 0 else 1
            
            if relative_multiple <= 0.8:
                score = 1.0
                signals.append({
                    'type': 'positive',
                    'text': f'Attractive at {multiple:.1f}x EBITDA ({(1-relative_multiple)*100:.0f}% below market)',
                    'impact': 'high'
                })
            elif relative_multiple <= 1.0:
                score = 0.7
                signals.append({
                    'type': 'positive',
                    'text': f'Fair value at {multiple:.1f}x EBITDA (market rate)',
                    'impact': 'medium'
                })
            elif relative_multiple <= 1.2:
                score = 0.4
                signals.append({
                    'type': 'neutral',
                    'text': f'Premium valuation at {multiple:.1f}x EBITDA ({(relative_multiple-1)*100:.0f}% above market)',
                    'impact': 'medium'
                })
            else:
                score = 0.1
                signals.append({
                    'type': 'warning',
                    'text': f'Expensive at {multiple:.1f}x EBITDA (consider negotiation)',
                    'impact': 'high'
                })
                
        return score, signals
    
    def _score_business_quality(self, company: Dict) -> Tuple[float, List[Dict]]:
        """Enhanced business quality assessment"""
        score = 0
        signals = []
        
        # Customer concentration with nuance
        customer_concentration = company.get('top_customer_concentration', 0)
        if customer_concentration < 15:
            score += 0.35
            signals.append({
                'type': 'positive',
                'text': 'Excellent customer diversification (<15% concentration)',
                'impact': 'high'
            })
        elif customer_concentration < 25:
            score += 0.25
            signals.append({
                'type': 'positive',
                'text': 'Good customer diversification',
                'impact': 'medium'
            })
        elif customer_concentration < 40:
            score += 0.1
            signals.append({
                'type': 'neutral',
                'text': f'Moderate customer concentration ({customer_concentration}%)',
                'impact': 'medium'
            })
        else:
            signals.append({
                'type': 'warning',
                'text': f'High customer concentration risk ({customer_concentration}%)',
                'impact': 'high'
            })
        
        # Business age and stability with context
        year_founded = company.get('year_founded', datetime.now().year)
        business_age = datetime.now().year - year_founded
        
        if business_age >= 15:
            score += 0.4
            signals.append({
                'type': 'positive',
                'text': f'Well-established business ({business_age} years) - proven model',
                'impact': 'high'
            })
        elif business_age >= 10:
            score += 0.3
            signals.append({
                'type': 'positive',
                'text': f'Established business ({business_age} years)',
                'impact': 'medium'
            })
        elif business_age >= 5:
            score += 0.15
            signals.append({
                'type': 'neutral',
                'text': f'Maturing business ({business_age} years)',
                'impact': 'low'
            })
        else:
            signals.append({
                'type': 'warning',
                'text': f'Young business ({business_age} years) - higher risk',
                'impact': 'medium'
            })
        
        # Market position with competitive analysis
        if company.get('market_leader', False):
            score += 0.3
            signals.append({
                'type': 'positive',
                'text': 'Market leader with defensible position',
                'impact': 'high'
            })
        elif company.get('market_position', '') == 'top_3':
            score += 0.2
            signals.append({
                'type': 'positive',
                'text': 'Top 3 player in market',
                'impact': 'medium'
            })
            
        # Intellectual property
        if company.get('patents_count', 0) > 0:
            score += 0.1
            signals.append({
                'type': 'positive',
                'text': f'{company["patents_count"]} patents provide competitive moat',
                'impact': 'medium'
            })
            
        return min(1.0, score), signals
    
    def _score_transition_ease(self, company: Dict) -> Tuple[float, List[Dict]]:
        """Enhanced transition ease evaluation"""
        score = 0
        signals = []
        
        # Management depth
        if company.get('has_management_team', False):
            if company.get('management_depth', 'thin') == 'strong':
                score += 0.6
                signals.append({
                    'type': 'positive',
                    'text': 'Deep management bench - smooth transition likely',
                    'impact': 'high'
                })
            else:
                score += 0.3
                signals.append({
                    'type': 'positive',
                    'text': 'Management team in place',
                    'impact': 'medium'
                })
        else:
            signals.append({
                'type': 'warning',
                'text': 'Owner-dependent operations - integration risk',
                'impact': 'high'
            })
        
        # Systems and processes maturity
        if company.get('documented_processes', False):
            if company.get('process_maturity', 'basic') == 'advanced':
                score += 0.35
                signals.append({
                    'type': 'positive',
                    'text': 'Mature, documented processes - plug-and-play opportunity',
                    'impact': 'high'
                })
            else:
                score += 0.2
                signals.append({
                    'type': 'positive',
                    'text': 'Documented processes exist',
                    'impact': 'medium'
                })
        
        # Technology infrastructure
        if company.get('modern_tech_stack', False):
            score += 0.15
            signals.append({
                'type': 'positive',
                'text': 'Modern technology stack - easier integration',
                'impact': 'medium'
            })
        
        # Seller cooperation
        if company.get('seller_will_stay', False):
            stay_duration = company.get('seller_stay_duration', 6)
            if stay_duration >= 12:
                score += 0.25
                signals.append({
                    'type': 'positive',
                    'text': f'Seller committed to {stay_duration}-month transition',
                    'impact': 'high'
                })
            else:
                score += 0.15
                signals.append({
                    'type': 'neutral',
                    'text': f'Seller available for {stay_duration}-month transition',
                    'impact': 'medium'
                })
                
        return min(1.0, score), signals


class AcquisitionMLPredictor:
    """
    Machine Learning component for acquisition success prediction
    """
    
    def __init__(self):
        self.model = self._initialize_model()
        self.feature_importance = self._load_feature_importance()
        
    def _initialize_model(self):
        """Initialize prediction model"""
        # In production, load trained model
        return {
            'type': 'gradient_boost',
            'version': '1.0',
            'accuracy': 0.82
        }
    
    def _load_feature_importance(self):
        """Load feature importance from trained model"""
        return {
            'owner_readiness': 0.25,
            'market_timing': 0.20,
            'financial_health': 0.20,
            'valuation_fairness': 0.15,
            'integration_ease': 0.10,
            'competitive_position': 0.10
        }
    
    def predict_success_probability(self, company: Dict, market_data: Dict) -> float:
        """Predict probability of successful acquisition"""
        features = self._extract_features(company, market_data)
        
        # Simplified prediction logic
        base_prob = 0.5
        
        # Adjust based on key factors
        if features['owner_age'] >= 60:
            base_prob += 0.15
        if features['ebitda_margin'] >= 25:
            base_prob += 0.1
        if features['market_consolidating']:
            base_prob += 0.1
        if features['fair_valuation']:
            base_prob += 0.1
        if features['has_management']:
            base_prob += 0.05
            
        return min(0.95, max(0.05, base_prob))
    
    def suggest_optimal_deal_structure(self, company: Dict) -> DealStructure:
        """AI-powered deal structure recommendation"""
        base_multiple = company.get('asking_price', 0) / max(company.get('ebitda', 1), 1)
        
        # Analyze risk factors
        risk_score = self._calculate_risk_score(company)
        
        # Adjust deal structure based on risk
        if risk_score < 0.3:  # Low risk
            suggested_multiple = base_multiple * 0.95
            earnout_pct = 10
            transition_months = 6
            confidence = 0.85
            rationale = "Low-risk acquisition with stable metrics"
        elif risk_score < 0.6:  # Medium risk
            suggested_multiple = base_multiple * 0.85
            earnout_pct = 25
            transition_months = 12
            confidence = 0.70
            rationale = "Moderate risk suggests performance-based structure"
        else:  # High risk
            suggested_multiple = base_multiple * 0.75
            earnout_pct = 40
            transition_months = 18
            confidence = 0.55
            rationale = "High-risk profile requires significant earnout protection"
            
        return DealStructure(
            suggested_multiple=suggested_multiple,
            earnout_percentage=earnout_pct,
            transition_months=transition_months,
            confidence_score=confidence,
            rationale=rationale
        )
    
    def _extract_features(self, company: Dict, market_data: Dict) -> Dict:
        """Extract ML features from company and market data"""
        return {
            'owner_age': company.get('owner_age', 50),
            'ebitda_margin': (company.get('ebitda', 0) / max(company.get('revenue', 1), 1)) * 100,
            'market_consolidating': market_data.get('consolidation_score', 0) > 0.6,
            'fair_valuation': self._is_valuation_fair(company),
            'has_management': company.get('has_management_team', False),
            'recurring_revenue': company.get('recurring_revenue_pct', 0) > 60
        }
    
    def _is_valuation_fair(self, company: Dict) -> bool:
        """Check if valuation is fair based on multiples"""
        if company.get('ebitda', 0) <= 0:
            return False
        multiple = company.get('asking_price', 0) / company['ebitda']
        return 3 <= multiple <= 6
    
    def _calculate_risk_score(self, company: Dict) -> float:
        """Calculate overall risk score"""
        risks = 0
        risk_factors = 0
        
        # Customer concentration risk
        if company.get('top_customer_concentration', 0) > 30:
            risks += 1
        risk_factors += 1
        
        # Owner dependency risk
        if not company.get('has_management_team', False):
            risks += 1
        risk_factors += 1
        
        # Technology risk
        if company.get('tech_stack_age', 0) > 5:
            risks += 0.5
        risk_factors += 1
        
        # Market risk
        if company.get('revenue_growth_rate', 0) < 5:
            risks += 0.5
        risk_factors += 1
        
        return risks / risk_factors if risk_factors > 0 else 0.5


class MarketIntelligence:
    """
    Real-time market intelligence for acquisition timing
    """
    
    def __init__(self):
        self.market_data = self._load_market_data()
        self.industry_trends = self._load_industry_trends()
        
    def _load_market_data(self):
        """Load current market conditions"""
        return {
            'interest_rates': 5.5,
            'credit_availability': 'moderate',
            'pe_dry_powder': 'high',
            'seller_sentiment': 'improving'
        }
    
    def _load_industry_trends(self):
        """Load industry-specific trends"""
        return {
            'SaaS': {
                'consolidation_rate': 0.8,
                'multiple_trend': 'stable',
                'buyer_competition': 'high'
            },
            'E-commerce': {
                'consolidation_rate': 0.6,
                'multiple_trend': 'declining',
                'buyer_competition': 'moderate'
            },
            'Manufacturing': {
                'consolidation_rate': 0.5,
                'multiple_trend': 'rising',
                'buyer_competition': 'low'
            }
        }
    
    def analyze_market_timing(self, company: Dict) -> Dict:
        """Comprehensive market timing analysis"""
        industry = company.get('industry', 'General')
        industry_data = self.industry_trends.get(industry, {})
        
        timing_factors = {
            'industry_consolidation': self._analyze_consolidation(industry_data),
            'valuation_cycle': self._analyze_valuation_cycle(industry_data),
            'competitive_dynamics': self._analyze_competition(industry_data),
            'macro_environment': self._analyze_macro_factors(),
            'seller_market_dynamics': self._analyze_seller_dynamics(company)
        }
        
        # Calculate composite timing score
        timing_score = sum(timing_factors.values()) / len(timing_factors)
        
        # Generate timing recommendation
        if timing_score >= 0.7:
            recommendation = {
                'action': 'BUY_NOW',
                'urgency': 'high',
                'rationale': 'Optimal market conditions for acquisition'
            }
        elif timing_score >= 0.5:
            recommendation = {
                'action': 'PREPARE_OFFER',
                'urgency': 'medium',
                'rationale': 'Good conditions, but some factors suggest patience'
            }
        else:
            recommendation = {
                'action': 'MONITOR',
                'urgency': 'low',
                'rationale': 'Market conditions not optimal, continue tracking'
            }
            
        return {
            'timing_score': timing_score,
            'factors': timing_factors,
            'recommendation': recommendation,
            'optimal_window': self._calculate_optimal_window(timing_score)
        }
    
    def get_comparable_transactions(self, company: Dict) -> List[Dict]:
        """Find comparable recent transactions"""
        # In production, this would query a transaction database
        industry = company.get('industry', 'General')
        revenue = company.get('revenue', 0)
        
        # Simulated comparable transactions
        comparables = [
            {
                'company': 'Similar Co 1',
                'date': '2024-10',
                'revenue': revenue * 0.9,
                'ebitda_multiple': 5.2,
                'strategic_buyer': True
            },
            {
                'company': 'Similar Co 2',
                'date': '2024-08',
                'revenue': revenue * 1.1,
                'ebitda_multiple': 4.8,
                'strategic_buyer': False
            }
        ]
        
        return comparables
    
    def _analyze_consolidation(self, industry_data: Dict) -> float:
        """Analyze industry consolidation trends"""
        return industry_data.get('consolidation_rate', 0.5)
    
    def _analyze_valuation_cycle(self, industry_data: Dict) -> float:
        """Analyze where valuations are in cycle"""
        trend = industry_data.get('multiple_trend', 'stable')
        if trend == 'declining':
            return 0.8  # Good time to buy
        elif trend == 'stable':
            return 0.6
        else:
            return 0.4  # Expensive
    
    def _analyze_competition(self, industry_data: Dict) -> float:
        """Analyze buyer competition"""
        competition = industry_data.get('buyer_competition', 'moderate')
        if competition == 'low':
            return 0.8
        elif competition == 'moderate':
            return 0.6
        else:
            return 0.4
    
    def _analyze_macro_factors(self) -> float:
        """Analyze macroeconomic factors"""
        # Consider interest rates, credit availability, etc.
        if self.market_data['interest_rates'] < 5:
            return 0.8
        elif self.market_data['interest_rates'] < 7:
            return 0.6
        else:
            return 0.4
    
    def _analyze_seller_dynamics(self, company: Dict) -> float:
        """Analyze seller-specific dynamics"""
        score = 0.5
        
        if company.get('actively_selling', False):
            score += 0.2
        if company.get('owner_age', 0) >= 60:
            score += 0.2
        if company.get('hired_investment_banker', False):
            score += 0.1
            
        return min(1.0, score)
    
    def _calculate_optimal_window(self, timing_score: float) -> str:
        """Calculate optimal acquisition window"""
        if timing_score >= 0.8:
            return "0-2 months (act quickly)"
        elif timing_score >= 0.6:
            return "2-4 months (prepare thoroughly)"
        elif timing_score >= 0.4:
            return "4-6 months (wait for better conditions)"
        else:
            return "6+ months (monitor only)"


class OutreachAutomation:
    """
    Automated outreach sequence generator
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        self.personalization_engine = self._initialize_personalization()
        
    def _load_templates(self):
        """Load outreach templates"""
        return {
            'initial_contact': {
                'subject_lines': [
                    "Quick question about {company}",
                    "Impressed by {company}'s growth",
                    "{owner_name}, exploring strategic options for {company}?"
                ],
                'opening_lines': [
                    "I've been following {company}'s impressive journey in the {industry} space",
                    "Your achievement of {key_metric} caught my attention",
                    "I noticed {recent_event} - congratulations on this milestone"
                ]
            },
            'follow_up': {
                'subject_lines': [
                    "Following up - partnership opportunity for {company}",
                    "Strategic alternatives for {company}",
                    "Quick thought on {company}'s next chapter"
                ]
            }
        }
    
    def _initialize_personalization(self):
        """Initialize personalization engine"""
        return {
            'tone_adjustment': True,
            'industry_specific': True,
            'owner_age_sensitive': True
        }
    
    def generate_outreach_sequence(self, company: Dict, score_data: Dict) -> Dict:
        """Generate personalized outreach sequence"""
        sequence = {
            'day_0': self._generate_initial_contact(company, score_data),
            'day_7': self._generate_first_followup(company, score_data),
            'day_14': self._generate_value_proposition(company, score_data),
            'day_30': self._generate_final_touch(company, score_data)
        }
        
        return sequence
    
    def _generate_initial_contact(self, company: Dict, score_data: Dict) -> Dict:
        """Generate initial outreach email"""
        # Select appropriate template based on score
        urgency = 'high' if score_data['score'] >= 80 else 'medium'
        
        # Personalize content
        owner_name = company.get('owner_name', 'Business Owner')
        company_name = company['company_name']
        industry = company.get('industry', 'your industry')
        
        # Find key achievement to mention
        key_metric = self._find_key_achievement(company)
        
        subject = f"Quick question about {company_name}"
        
        body = f"""Hi {owner_name},

I've been following {company_name}'s impressive journey in the {industry} space, particularly your {key_metric}.

I lead acquisitions at Caprae Capital, where we partner with successful founders to provide liquidity while preserving their legacy. We're particularly interested in businesses like yours that have built something special.

Would you be open to a brief confidential conversation about your long-term vision for {company_name}? Even if the timing isn't right today, I'd value the opportunity to learn about your business and share how we've helped similar founders achieve their goals.

Best regards,
[Your Name]
Caprae Capital Partners
"""
        
        return {
            'subject': subject,
            'body': body,
            'urgency': urgency,
            'personalization_score': 0.85
        }
    
    def _find_key_achievement(self, company: Dict) -> str:
        """Find the most impressive metric to mention"""
        achievements = []
        
        # Check various metrics
        if company.get('revenue_growth_rate', 0) > 25:
            achievements.append(f"{company['revenue_growth_rate']}% annual growth")
            
        if company.get('ebitda', 0) / max(company.get('revenue', 1), 1) > 0.25:
            achievements.append("exceptional profitability")
            
        if company.get('employees', 0) > 100:
            achievements.append(f"growth to {company['employees']} employees")
            
        if company.get('years_in_business', 0) > 15:
            achievements.append(f"{company['years_in_business']} years of success")
            
        return achievements[0] if achievements else "consistent growth"
    
    def _generate_first_followup(self, company: Dict, score_data: Dict) -> Dict:
        """Generate first follow-up email"""
        company_name = company['company_name']
        
        body = f"""Hi {company.get('owner_name', 'there')},

I wanted to follow up on my previous note about {company_name}. 

I realize you're busy running a successful business, so I'll be brief. We recently completed an acquisition of a similar {company.get('industry', 'company')} that resulted in:

• 40% increase in enterprise value within 18 months
• Founder retained 20% equity and board seat
• Management team received retention bonuses and growth incentives

If you're curious about how a partnership might look for {company_name}, I'd be happy to share a confidential preliminary valuation range based on your metrics.

Best regards,
[Your Name]
"""
        
        return {
            'subject': f"Following up - strategic options for {company_name}",
            'body': body
        }
    
    def _generate_value_proposition(self, company: Dict, score_data: Dict) -> Dict:
        """Generate value proposition email"""
        return {
            'subject': f"One specific idea for {company['company_name']}'s growth",
            'body': self._create_value_prop_body(company, score_data)
        }
    
    def _generate_final_touch(self, company: Dict, score_data: Dict) -> Dict:
        """Generate final touch email"""
        return {
            'subject': "Last note - door remains open",
            'body': self._create_final_touch_body(company)
        }
    
    def _create_value_prop_body(self, company: Dict, score_data: Dict) -> str:
        """Create value proposition email body"""
        company_name = company['company_name']
        industry = company.get('industry', 'your industry')
        
        return f"""Hi {company.get('owner_name', 'there')},

I've been thinking about {company_name} and wanted to share one specific opportunity I see:

Based on our analysis of the {industry} market, companies with your profile are perfectly positioned for roll-up strategies. We've identified 3-4 smaller players that could be acquired and integrated, potentially doubling your market share within 24 months.

This isn't just theory - we executed a similar strategy with [Portfolio Company] last year, resulting in a 3.5x return for the founder.

If growth through acquisition is something you've considered, I'd love to share our specific thesis for {company_name}.

Best regards,
[Your Name]
"""
    
    def _create_final_touch_body(self, company: Dict) -> str:
        """Create final touch email body"""
        return f"""Hi {company.get('owner_name', 'there')},

I wanted to reach out one last time regarding {company['company_name']}.

I understand the timing may not be right for a conversation, and I respect that. Building a great business takes focus, and I'm sure you have plenty on your plate.

If circumstances change in the future, please don't hesitate to reach out. We maintain relationships with founders for years before the timing aligns, and our door is always open.

Wishing you continued success with {company['company_name']}.

Best regards,
[Your Name]

P.S. - If you know other founders who might be exploring strategic alternatives, I'd be grateful for an introduction.
"""


class DataEnricher:
    """
    Enhanced data enrichment with AI-powered insights
    """
    
    def __init__(self):
        self.enrichment_sources = self._initialize_sources()
        self.ai_analyzer = self._initialize_ai_analyzer()
        
    def _initialize_sources(self):
        """Initialize data sources for enrichment"""
        return {
            'business_signals': True,
            'financial_estimates': True,
            'market_intelligence': True,
            'social_signals': True,
            'news_sentiment': True
        }
    
    def _initialize_ai_analyzer(self):
        """Initialize AI analysis capabilities"""
        return {
            'pattern_recognition': True,
            'anomaly_detection': True,
            'predictive_modeling': True
        }
    
    def enrich_company_data(self, company_name: str, website: str, 
                           basic_data: Dict) -> Dict:
        """
        Comprehensive data enrichment with AI insights
        """
        enriched = {
            'company_name': company_name,
            'website': website,
            'enrichment_timestamp': datetime.now().isoformat(),
            'confidence_score': 0.0
        }
        
        # Layer 1: Basic enrichment
        enriched.update(self._basic_enrichment(company_name, website, basic_data))
        
        # Layer 2: Signal detection
        enriched['signals'] = self._detect_acquisition_signals(enriched)
        
        # Layer 3: AI analysis
        enriched['ai_insights'] = self._generate_ai_insights(enriched)
        
        # Layer 4: Risk assessment
        enriched['risk_profile'] = self._assess_risks(enriched)
        
        # Layer 5: Opportunity scoring
        enriched['hidden_opportunities'] = self._find_hidden_opportunities(enriched)
        
        # Calculate overall confidence
        enriched['confidence_score'] = self._calculate_confidence(enriched)
        
        return enriched
    
    def _basic_enrichment(self, company_name: str, website: str, 
                         basic_data: Dict) -> Dict:
        """Basic data enrichment"""
        return {
            'basic_info': {
                'employees': basic_data.get('employees', 50),
                'revenue_estimate': self._estimate_revenue(basic_data),
                'growth_stage': self._determine_growth_stage(basic_data),
                'technology_stack': self._detect_tech_stack(website),
                'social_presence': self._analyze_social_presence(company_name)
            },
            'owner_signals': self._get_owner_signals(company_name, basic_data),
            'financial_estimates': self._estimate_detailed_financials(basic_data),
            'market_position': self._analyze_market_position(company_name, basic_data)
        }
    
    def _detect_acquisition_signals(self, enriched_data: Dict) -> List[Dict]:
        """Detect signals indicating acquisition readiness"""
        signals = []
        
        # Owner-related signals
        owner_data = enriched_data.get('owner_signals', {})
        if owner_data.get('age_estimate', 0) >= 55:
            signals.append({
                'type': 'owner_readiness',
                'strength': 'strong',
                'description': 'Owner in typical exit age range',
                'confidence': 0.8
            })
            
        if owner_data.get('succession_mentioned', False):
            signals.append({
                'type': 'succession_planning',
                'strength': 'strong',
                'description': 'Succession planning mentioned publicly',
                'confidence': 0.9
            })
            
        # Financial signals
        financial_data = enriched_data.get('financial_estimates', {})
        if financial_data.get('growth_slowing', False):
            signals.append({
                'type': 'growth_plateau',
                'strength': 'medium',
                'description': 'Growth rate appears to be slowing',
                'confidence': 0.7
            })
            
        # Market signals
        if enriched_data.get('market_position', {}).get('consolidation_target', False):
            signals.append({
                'type': 'market_consolidation',
                'strength': 'strong',
                'description': 'Industry consolidation makes company attractive target',
                'confidence': 0.85
            })
            
        return signals
    
    def _generate_ai_insights(self, enriched_data: Dict) -> Dict:
        """Generate AI-powered insights"""
        insights = {
            'readiness_prediction': self._predict_exit_readiness(enriched_data),
            'optimal_approach': self._suggest_approach(enriched_data),
            'value_creation_opportunities': self._identify_value_creation(enriched_data),
            'integration_complexity': self._assess_integration_complexity(enriched_data)
        }
        
        return insights
    
    def _find_hidden_opportunities(self, enriched_data: Dict) -> List[str]:
        """Find non-obvious value creation opportunities"""
        opportunities = []
        
        # Technology modernization
        tech_stack = enriched_data.get('basic_info', {}).get('technology_stack', [])
        if self._is_tech_outdated(tech_stack):
            opportunities.append("Technology modernization could unlock 20-30% efficiency gains")
            
        # Market expansion
        if enriched_data.get('market_position', {}).get('regional_player', False):
            opportunities.append("Geographic expansion opportunity - currently regional player")
            
        # Pricing optimization
        if enriched_data.get('financial_estimates', {}).get('below_market_pricing', False):
            opportunities.append("Pricing optimization could improve margins by 5-10%")
            
        # Cross-sell opportunities
        if enriched_data.get('basic_info', {}).get('single_product_focus', False):
            opportunities.append("Product line expansion through cross-selling")
            
        return opportunities
    
    # Helper methods for enrichment
    def _estimate_revenue(self, basic_data: Dict) -> Tuple[int, int]:
        """Estimate revenue range based on employee count and industry"""
        employees = basic_data.get('employees', 50)
        industry = basic_data.get('industry', 'General')
        
        # Industry-specific revenue per employee
        revenue_per_employee = {
            'SaaS': 200000,
            'E-commerce': 150000,
            'Manufacturing': 300000,
            'Healthcare': 180000,
            'Professional Services': 150000
        }
        
        rpe = revenue_per_employee.get(industry, 175000)
        base_revenue = employees * rpe
        
        # Return range
        return (int(base_revenue * 0.8), int(base_revenue * 1.2))
    
    def _determine_growth_stage(self, basic_data: Dict) -> str:
        """Determine company's growth stage"""
        employees = basic_data.get('employees', 0)
        years = datetime.now().year - basic_data.get('year_founded', datetime.now().year)
        
        if employees < 20 or years < 3:
            return 'early_stage'
        elif employees < 50 or years < 7:
            return 'growth_stage'
        elif employees < 200 or years < 15:
            return 'expansion_stage'
        else:
            return 'mature_stage'
    
    def _detect_tech_stack(self, website: str) -> List[str]:
        """Detect technology stack (simplified)"""
        # In production, would actually analyze the website
        return ['React', 'AWS', 'PostgreSQL', 'Python']
    
    def _analyze_social_presence(self, company_name: str) -> Dict:
        """Analyze social media presence"""
        # Simplified - in production would check actual social APIs
        return {
            'linkedin_employees': random.randint(20, 200),
            'linkedin_followers': random.randint(500, 5000),
            'twitter_followers': random.randint(100, 2000),
            'engagement_rate': random.uniform(0.02, 0.05)
        }
    
    def _get_owner_signals(self, company_name: str, basic_data: Dict) -> Dict:
        """Detect owner-related signals"""
        # In production: LinkedIn analysis, news search, etc.
        return {
            'age_estimate': random.randint(45, 65),
            'succession_mentioned': random.choice([True, False]),
            'advisor_hires': random.choice([
                [],
                ['Investment banker engaged'],
                ['M&A attorney hired', 'Business broker consulted']
            ]),
            'linkedin_activity': random.choice(['increasing', 'stable', 'decreasing'])
        }
    
    def _estimate_detailed_financials(self, basic_data: Dict) -> Dict:
        """Estimate detailed financial metrics"""
        revenue_range = self._estimate_revenue(basic_data)
        avg_revenue = (revenue_range[0] + revenue_range[1]) / 2
        
        # Industry-based margin estimates
        industry = basic_data.get('industry', 'General')
        margin_ranges = {
            'SaaS': (0.15, 0.35),
            'E-commerce': (0.08, 0.20),
            'Manufacturing': (0.10, 0.25),
            'Healthcare': (0.12, 0.28),
            'Professional Services': (0.15, 0.30)
        }
        
        margins = margin_ranges.get(industry, (0.10, 0.25))
        ebitda_margin = random.uniform(margins[0], margins[1])
        
        return {
            'revenue_range': revenue_range,
            'ebitda_estimate': int(avg_revenue * ebitda_margin),
            'ebitda_margin_estimate': ebitda_margin,
            'growth_estimate': random.uniform(5, 35),
            'recurring_revenue_estimate': random.uniform(20, 90),
            'growth_slowing': random.choice([True, False]),
            'below_market_pricing': random.choice([True, False])
        }
    
    def _analyze_market_position(self, company_name: str, basic_data: Dict) -> Dict:
        """Analyze company's market position"""
        return {
            'market_share_estimate': random.uniform(0.05, 0.25),
            'competitive_strength': random.choice(['leader', 'strong', 'average', 'weak']),
            'consolidation_target': random.choice([True, False]),
            'regional_player': random.choice([True, False]),
            'defensible_moat': random.choice([True, False])
        }
    
    def _predict_exit_readiness(self, enriched_data: Dict) -> Dict:
        """Predict likelihood of exit readiness"""
        score = 0.5  # Base score
        
        # Adjust based on signals
        owner_age = enriched_data.get('owner_signals', {}).get('age_estimate', 50)
        if owner_age >= 60:
            score += 0.2
        elif owner_age >= 55:
            score += 0.1
            
        if enriched_data.get('financial_estimates', {}).get('growth_slowing', False):
            score += 0.1
            
        if enriched_data.get('owner_signals', {}).get('advisor_hires', []):
            score += 0.15
            
        return {
            'probability': min(0.95, score),
            'timeframe': '6-12 months' if score > 0.7 else '12-24 months',
            'confidence': 0.75
        }
    
    def _suggest_approach(self, enriched_data: Dict) -> str:
        """Suggest optimal approach strategy"""
        owner_age = enriched_data.get('owner_signals', {}).get('age_estimate', 50)
        growth_stage = enriched_data.get('basic_info', {}).get('growth_stage', 'unknown')
        
        if owner_age >= 60 and growth_stage == 'mature_stage':
            return 'direct_approach_retirement_focus'
        elif growth_stage == 'expansion_stage':
            return 'growth_partnership_approach'
        else:
            return 'relationship_building_approach'
    
    def _identify_value_creation(self, enriched_data: Dict) -> List[str]:
        """Identify value creation opportunities"""
        opportunities = []
        
        # Operational improvements
        if enriched_data.get('financial_estimates', {}).get('ebitda_margin_estimate', 0) < 0.20:
            opportunities.append('Operational efficiency improvements')
            
        # Technology upgrades
        tech_stack = enriched_data.get('basic_info', {}).get('technology_stack', [])
        if self._is_tech_outdated(tech_stack):
            opportunities.append('Technology infrastructure modernization')
            
        # Market expansion
        if enriched_data.get('market_position', {}).get('regional_player', False):
            opportunities.append('Geographic market expansion')
            
        return opportunities
    
    def _assess_integration_complexity(self, enriched_data: Dict) -> str:
        """Assess how complex integration would be"""
        complexity_score = 0
        
        # Technology complexity
        tech_stack = enriched_data.get('basic_info', {}).get('technology_stack', [])
        if len(tech_stack) > 5 or self._is_tech_outdated(tech_stack):
            complexity_score += 1
            
        # Organizational complexity
        employees = enriched_data.get('basic_info', {}).get('employees', 0)
        if employees > 100:
            complexity_score += 1
            
        # Market complexity
        if enriched_data.get('market_position', {}).get('regional_player', False):
            complexity_score += 1
            
        if complexity_score >= 2:
            return 'high'
        elif complexity_score == 1:
            return 'medium'
        else:
            return 'low'
    
    def _is_tech_outdated(self, tech_stack: List[str]) -> bool:
        """Check if technology stack is outdated"""
        outdated_indicators = ['PHP', 'jQuery', 'MySQL', 'Windows Server']
        return any(tech in tech_stack for tech in outdated_indicators)
    
    def _assess_risks(self, enriched_data: Dict) -> Dict:
        """Comprehensive risk assessment"""
        risks = []
        
        # Customer concentration risk
        if random.random() > 0.5:  # Simplified - would check actual data
            risks.append({
                'type': 'customer_concentration',
                'severity': 'high',
                'mitigation': 'Diversify customer base post-acquisition'
            })
            
        # Technology risk
        if self._is_tech_outdated(enriched_data.get('basic_info', {}).get('technology_stack', [])):
            risks.append({
                'type': 'technology_debt',
                'severity': 'medium',
                'mitigation': 'Budget for technology modernization'
            })
            
        # Key person risk
        if not enriched_data.get('basic_info', {}).get('has_management_team', False):
            risks.append({
                'type': 'key_person_dependency',
                'severity': 'high',
                'mitigation': 'Retention agreements and management hiring'
            })
            
        overall_risk = 'high' if len([r for r in risks if r['severity'] == 'high']) >= 2 else 'medium'
        
        return {
            'overall_risk': overall_risk,
            'risk_factors': risks,
            'risk_score': len(risks) / 5.0  # Normalized score
        }
    
    def _calculate_confidence(self, enriched_data: Dict) -> float:
        """Calculate overall confidence in enrichment data"""
        confidence_factors = []
        
        # Data completeness
        required_fields = ['basic_info', 'owner_signals', 'financial_estimates', 'market_position']
        completeness = sum(1 for field in required_fields if field in enriched_data) / len(required_fields)
        confidence_factors.append(completeness)
        
        # Signal strength
        signals = enriched_data.get('signals', [])
        if signals:
            strong_signals = [s for s in signals if s.get('strength') == 'strong']
            signal_strength = len(strong_signals) / max(len(signals), 1)
            confidence_factors.append(signal_strength)
        
        # AI insights confidence
        ai_insights = enriched_data.get('ai_insights', {})
        if ai_insights:
            readiness_conf = ai_insights.get('readiness_prediction', {}).get('confidence', 0)
            confidence_factors.append(readiness_conf)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5


# API Endpoint
class AcquisitionAPI:
    """
    RESTful API for acquisition scoring
    """
    
    def __init__(self):
        self.scorer = AcquisitionScorer()
        self.enricher = DataEnricher()
        self.ml_predictor = AcquisitionMLPredictor()
        self.market_intel = MarketIntelligence()
        self.outreach = OutreachAutomation()
        
    def score_company_endpoint(self, company_data: Dict) -> Dict:
        """
        Main API endpoint for company scoring
        """
        try:
            # Enrich the data
            enriched = self.enricher.enrich_company_data(
                company_data.get('company_name', ''),
                company_data.get('website', ''),
                company_data
            )
            
            # Merge enriched data with original
            company_data.update(enriched['financial_estimates'])
            company_data.update(enriched['owner_signals'])
            
            # Score with AI
            score, signals, ai_insights = self.scorer.score_target_with_ai(company_data)
            
            # Get ML predictions
            success_probability = self.ml_predictor.predict_success_probability(
                company_data, 
                ai_insights['market_timing']
            )
            
            # Get optimal deal structure
            deal_structure = self.ml_predictor.suggest_optimal_deal_structure(company_data)
            
            # Market timing analysis
            market_timing = self.market_intel.analyze_market_timing(company_data)
            
            # Generate outreach sequence
            outreach_sequence = self.outreach.generate_outreach_sequence(
                company_data, 
                {'score': score, 'ai_insights': ai_insights}
            )
            
            # Compile response
            response = {
                'api_version': '2.0',
                'timestamp': datetime.now().isoformat(),
                'company': {
                    'name': company_data.get('company_name'),
                    'website': company_data.get('website'),
                    'industry': company_data.get('industry')
                },
                'acquisition_score': {
                    'total_score': score,
                    'rating': self._get_rating(score),
                    'confidence': enriched['confidence_score']
                },
                'signals': signals[:5],  # Top 5 signals
                'ai_insights': {
                    'success_probability': round(success_probability, 2),
                    'market_timing': market_timing['recommendation'],
                    'optimal_window': market_timing['optimal_window'],
                    'hidden_opportunities': ai_insights['hidden_opportunities'][:3]
                },
                'deal_recommendation': {
                    'suggested_multiple': round(deal_structure.suggested_multiple, 1),
                    'earnout_percentage': deal_structure.earnout_percentage,
                    'transition_months': deal_structure.transition_months,
                    'structure_confidence': round(deal_structure.confidence_score, 2),
                    'rationale': deal_structure.rationale
                },
                'next_steps': self._get_next_steps(score, ai_insights),
                'outreach_preview': {
                    'initial_email_subject': outreach_sequence['day_0']['subject'],
                    'approach_strategy': enriched['ai_insights']['optimal_approach']
                },
                'integration_ready': True
            }
            
            return response
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error',
                'api_version': '2.0'
            }
    
    def _get_rating(self, score: int) -> str:
        """Convert score to rating"""
        if score >= 80:
            return 'HOT - Immediate Action Required'
        elif score >= 60:
            return 'WARM - Strong Opportunity'
        elif score >= 40:
            return 'COOL - Monitor and Nurture'
        else:
            return 'COLD - Long-term Tracking'
    
    def _get_next_steps(self, score: int, ai_insights: Dict) -> List[str]:
        """Generate recommended next steps"""
        steps = []
        
        if score >= 80:
            steps.extend([
                "Initiate immediate outreach to owner",
                "Prepare preliminary valuation model",
                "Research comparable transactions",
                "Schedule internal investment committee preview"
            ])
        elif score >= 60:
            steps.extend([
                "Add to high-priority pipeline",
                "Begin relationship building with owner",
                "Conduct deeper market analysis",
                "Monitor for additional readiness signals"
            ])
        else:
            steps.extend([
                "Add to long-term tracking list",
                "Set up quarterly check-ins",
                "Monitor for trigger events",
                "Build industry relationships"
            ])
            
        return steps[:4]  # Return top 4 steps


# Enhanced Data Generation
def generate_enhanced_sample_data():
    """Generate comprehensive sample data with AI-relevant fields"""
    
    companies = [
        {
            # Basic info
            'company_name': 'CloudScale Analytics',
            'website': 'cloudscaleanalytics.com',
            'industry': 'SaaS',
            'location': 'Austin, TX',
            'employees': 125,
            'year_founded': 2014,
            
            # Financial data
            'revenue': 22000000,
            'ebitda': 5500000,
            'revenue_growth_rate': 35,
            'recurring_revenue_pct': 92,
            'growth_trend': 'accelerating',
            
            # Owner info
            'owner_age': 58,
            'owner_name': 'Michael Chen',
            'years_owned': 10,
            'actively_selling': False,
            'hired_investment_banker': True,
            'family_business': False,
            
            # Valuation
            'asking_price': 88000000,
            
            # Quality indicators
            'top_customer_concentration': 12,
            'has_management_team': True,
            'management_depth': 'strong',
            'documented_processes': True,
            'process_maturity': 'advanced',
            'modern_tech_stack': True,
            'seller_will_stay': True,
            'seller_stay_duration': 12,
            'market_leader': True,
            'patents_count': 3,
            
            # Additional AI fields
            'tech_stack_age': 2,
            'r_and_d_spend': 3300000,
            'customer_retention': 96,
            'employee_tenure_avg': 4.5,
            'market_share_trend': 'growing'
        },
        {
            'company_name': 'MedConnect Solutions',
            'website': 'medconnectsolutions.com',
            'industry': 'Healthcare',
            'location': 'Boston, MA',
            'employees': 200,
            'year_founded': 2010,
            'revenue': 45000000,
            'ebitda': 9000000,
            'revenue_growth_rate': 18,
            'recurring_revenue_pct': 78,
            'growth_trend': 'stable',
            'owner_age': 62,
            'owner_name': 'Dr. Sarah Williams',
            'years_owned': 14,
            'actively_selling': True,
            'hired_investment_banker': False,
            'family_business': True,
            'asking_price': 54000000,
            'top_customer_concentration': 25,
            'has_management_team': True,
            'management_depth': 'thin',
            'documented_processes': True,
            'process_maturity': 'basic',
            'modern_tech_stack': False,
            'seller_will_stay': True,
            'seller_stay_duration': 18,
            'market_position': 'top_3',
            'patents_count': 8,
            'tech_stack_age': 6,
            'r_and_d_spend': 2250000,
            'customer_retention': 91,
            'employee_tenure_avg': 6.2,
            'market_share_trend': 'stable'
        },
        {
            'company_name': 'EcoShip Logistics',
            'website': 'ecoshiplogistics.com',
            'industry': 'E-commerce',
            'location': 'Miami, FL',
            'employees': 85,
            'year_founded': 2016,
            'revenue': 15000000,
            'ebitda': 2250000,
            'revenue_growth_rate': 45,
            'recurring_revenue_pct': 35,
            'growth_trend': 'accelerating',
            'owner_age': 42,
            'owner_name': 'Carlos Rodriguez',
            'years_owned': 8,
            'actively_selling': False,
            'hired_investment_banker': False,
            'family_business': False,
            'asking_price': 18000000,
            'top_customer_concentration': 55,
            'has_management_team': False,
            'documented_processes': False,
            'modern_tech_stack': True,
            'seller_will_stay': False,
            'market_leader': False,
            'patents_count': 0,
            'tech_stack_age': 1,
            'r_and_d_spend': 750000,
            'customer_retention': 82,
            'employee_tenure_avg': 2.5,
            'market_share_trend': 'growing'
        },
        {
            'company_name': 'Precision Manufacturing Co',
            'website': 'precisionmfgco.com',
            'industry': 'Manufacturing',
            'location': 'Cleveland, OH',
            'employees': 250,
            'year_founded': 1995,
            'revenue': 48000000,
            'ebitda': 8640000,
            'revenue_growth_rate': 8,
            'recurring_revenue_pct': 65,
            'growth_trend': 'stable',
            'owner_age': 67,
            'owner_name': 'Robert Anderson',
            'years_owned': 28,
            'actively_selling': True,
            'hired_investment_banker': True,
            'family_business': True,
            'asking_price': 52000000,
            'top_customer_concentration': 30,
            'has_management_team': True,
            'management_depth': 'strong',
            'documented_processes': True,
            'process_maturity': 'advanced',
            'modern_tech_stack': False,
            'seller_will_stay': True,
            'seller_stay_duration': 6,
            'market_leader': False,
            'market_position': 'top_3',
            'patents_count': 12,
            'tech_stack_age': 8,
            'r_and_d_spend': 1440000,
            'customer_retention': 94,
            'employee_tenure_avg': 12.5,
            'market_share_trend': 'stable'
        },
        {
            'company_name': 'Strategic Advisors Group',
            'website': 'strategicadvisorsgroup.com',
            'industry': 'Professional Services',
            'location': 'Chicago, IL',
            'employees': 175,
            'year_founded': 2008,
            'revenue': 35000000,
            'ebitda': 7700000,
            'revenue_growth_rate': 22,
            'recurring_revenue_pct': 45,
            'growth_trend': 'stable',
            'owner_age': 54,
            'owner_name': 'Jennifer Park',
            'years_owned': 16,
            'actively_selling': False,
            'hired_investment_banker': False,
            'family_business': False,
            'asking_price': 46200000,
            'top_customer_concentration': 20,
            'has_management_team': True,
            'management_depth': 'strong',
            'documented_processes': True,
            'process_maturity': 'basic',
            'modern_tech_stack': True,
            'seller_will_stay': True,
            'seller_stay_duration': 24,
            'market_leader': False,
            'patents_count': 0,
            'tech_stack_age': 3,
            'r_and_d_spend': 0,
            'customer_retention': 89,
            'employee_tenure_avg': 5.8,
            'market_share_trend': 'growing'
        }
    ]
    
    return companies


def main():
    """Enhanced main execution with AI capabilities"""
    
    print("=" * 70)
    print("SAASQUATCH AI-POWERED ACQUISITION INTELLIGENCE ENGINE")
    print("=" * 70)
    print()
    
    # Initialize all components
    scorer = AcquisitionScorer()
    enricher = DataEnricher()
    ml_predictor = AcquisitionMLPredictor()
    market_intel = MarketIntelligence()
    outreach = OutreachAutomation()
    api = AcquisitionAPI()
    
    # Get enhanced sample data
    companies = generate_enhanced_sample_data()
    
    # Process each company with full AI analysis
    results = []
    
    for company in companies:
        print(f"Analyzing {company['company_name']}...")
        
        # Use API endpoint for comprehensive analysis
        api_result = api.score_company_endpoint(company)
        
        # Store results
        results.append({
            'company': company,
            'api_result': api_result,
            'score': api_result['acquisition_score']['total_score'],
            'success_probability': api_result['ai_insights']['success_probability'],
            'timing': api_result['ai_insights']['market_timing'],
            'deal_multiple': api_result['deal_recommendation']['suggested_multiple']
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Output comprehensive results
    print("\n" + "=" * 70)
    print("TOP ACQUISITION TARGETS - AI ANALYSIS")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        company = result['company']
        api_data = result['api_result']
        
        print(f"\n{i}. {company['company_name']} - {api_data['acquisition_score']['rating']}")
        print(f"   Score: {api_data['acquisition_score']['total_score']} | "
              f"Success Probability: {api_data['ai_insights']['success_probability']*100:.0f}%")
        print(f"   Industry: {company['industry']} | Location: {company['location']}")
        print(f"   Revenue: ${company['revenue']:,.0f} | EBITDA: ${company['ebitda']:,.0f}")
        print(f"   Current Ask: ${company['asking_price']:,.0f} ({company['asking_price']/company['ebitda']:.1f}x)")
        print(f"   AI Suggested: ${company['ebitda'] * api_data['deal_recommendation']['suggested_multiple']:,.0f} "
              f"({api_data['deal_recommendation']['suggested_multiple']:.1f}x)")
        print(f"   Deal Structure: {api_data['deal_recommendation']['earnout_percentage']}% earnout, "
              f"{api_data['deal_recommendation']['transition_months']} month transition")
        
        print(f"\n   Market Timing: {api_data['ai_insights']['market_timing']['action']}")
        print(f"   Optimal Window: {api_data['ai_insights']['optimal_window']}")
        
        print("\n   Top Signals:")
        for signal in api_data['signals'][:3]:
            confidence = signal.get('confidence', 0.5)
            print(f"   • {signal['text']} (confidence: {confidence:.0%})")
        
        print("\n   Hidden Opportunities:")
        for opp in api_data['ai_insights']['hidden_opportunities'][:2]:
            print(f"   • {opp}")
        
        print("\n   Recommended Next Steps:")
        for step in api_data['next_steps'][:3]:
            print(f"   ✓ {step}")
        
        print("\n" + "-" * 70)
    
    # Summary statistics
    print(f"\nPORTFOLIO SUMMARY:")
    print(f"• Total Targets Analyzed: {len(results)}")
    print(f"• Hot Leads (80+): {len([r for r in results if r['score'] >= 80])}")
    print(f"• Average Success Probability: {np.mean([r['success_probability'] for r in results]):.0%}")
    print(f"• Immediate Action Required: {len([r for r in results if r['timing']['action'] == 'BUY_NOW'])}")
    
    # API demonstration
    print("\n" + "=" * 70)
    print("API ENDPOINT DEMONSTRATION")
    print("=" * 70)
    print("\nExample API Response for Top Target:")
    print(json.dumps(results[0]['api_result'], indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
