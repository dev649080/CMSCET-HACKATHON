
import { supabase } from './supabase';

export interface ProcessReading {
  id: string;
  timestamp: Date;
  furnace_id: string;
  temperature: number;
  pressure: number;
  oxygen_level: number;
  composition: Record<string, number>;
  quality_score?: number;
}

export interface AlloyRecommendation {
  id: string;
  target_composition: Record<string, number>;
  current_composition: Record<string, number>;
  recommendations: Array<{
    element: string;
    adjustment: number;
    confidence: number;
  }>;
  cost_impact: number;
  quality_improvement: number;
  created_at: Date;
}

export interface Alert {
  id: string;
  title: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  is_resolved: boolean;
  created_at: Date;
  resolved_at?: Date;
}

class DataService {
  // Process Data Management
  async getRecentProcessData(hours: number = 24): Promise<ProcessReading[]> {
    const cutoffTime = new Date(Date.now() - hours * 60 * 60 * 1000);

    const { data, error } = await supabase
      .from('process_data')
      .select('*')
      .gte('timestamp', cutoffTime.toISOString())
      .order('timestamp', { ascending: false });

    if (error) {
      console.error('Error fetching process data from database:', error);
      // Throw error instead of returning mock data
      throw new Error('Failed to fetch process data. Please try again.');
    }

    if (!data || data.length === 0) {
      console.warn('No process data available for the requested period');
      return [];
    }

    return data;
  }

  async addProcessReading(reading: Omit<ProcessReading, 'id'>): Promise<void> {
    const { error } = await supabase
      .from('process_data')
      .insert([{
        timestamp: reading.timestamp.toISOString(),
        furnace_id: reading.furnace_id,
        temperature: reading.temperature,
        pressure: reading.pressure,
        oxygen_level: reading.oxygen_level,
        composition_data: reading.composition,
        quality_score: reading.quality_score
      }]);

    if (error) {
      console.error('Error adding process reading:', error);
    }
  }

  // Alloy Recommendations
  async getRecommendations(): Promise<AlloyRecommendation[]> {
    const { data, error } = await supabase
      .from('alloy_recommendations')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(10);

    if (error) {
      console.error('Error fetching recommendations from database:', error);
      throw new Error('Failed to fetch recommendations. Please try again.');
    }

    if (!data || data.length === 0) {
      console.warn('No recommendations available yet');
      return [];
    }

    return data;
  }

  async generateRecommendation(
    targetComposition: Record<string, number>,
    currentComposition: Record<string, number>
  ): Promise<AlloyRecommendation> {
    // Simulate AI recommendation generation
    const recommendations = Object.keys(targetComposition).map(element => {
      const target = targetComposition[element];
      const current = currentComposition[element] || 0;
      const adjustment = target - current;
      const confidence = Math.random() * 20 + 80; // 80-100% confidence

      return {
        element,
        adjustment,
        confidence
      };
    });

    const costImpact = recommendations.reduce((sum, rec) =>
      sum + Math.abs(rec.adjustment) * 10, 0
    );

    const qualityImprovement = Math.random() * 15 + 5; // 5-20% improvement

    const recommendation: AlloyRecommendation = {
      id: crypto.randomUUID(),
      target_composition: targetComposition,
      current_composition: currentComposition,
      recommendations,
      cost_impact: costImpact,
      quality_improvement: qualityImprovement,
      created_at: new Date()
    };

    // Store in database
    const { error } = await supabase
      .from('alloy_recommendations')
      .insert([{
        target_composition: targetComposition,
        current_composition: currentComposition,
        recommendations: recommendations,
        cost_impact: costImpact,
        quality_improvement: qualityImprovement
      }]);

    if (error) {
      console.error('Error storing recommendation:', error);
    }

    return recommendation;
  }

  // Alert Management
  async getActiveAlerts(): Promise<Alert[]> {
    const { data, error } = await supabase
      .from('alerts')
      .select('*')
      .eq('is_resolved', false)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching alerts from database:', error);
      throw new Error('Failed to fetch alerts. Please try again.');
    }

    if (!data || data.length === 0) {
      console.warn('No active alerts');
      return [];
    }

    return data;
  }

  async createAlert(alert: Omit<Alert, 'id' | 'created_at' | 'is_resolved'>): Promise<void> {
    const { error } = await supabase
      .from('alerts')
      .insert([{
        title: alert.title,
        message: alert.message,
        severity: alert.severity,
        source: alert.source,
        is_resolved: false
      }]);

    if (error) {
      console.error('Error creating alert:', error);
    }
  }

  async resolveAlert(alertId: string): Promise<void> {
    const { error } = await supabase
      .from('alerts')
      .update({
        is_resolved: true,
        resolved_at: new Date().toISOString()
      })
      .eq('id', alertId);

    if (error) {
      console.error('Error resolving alert:', error);
    }
  }

  // Analytics and Reporting
  async getSystemAnalytics() {
    const [processData, alerts, recommendations] = await Promise.all([
      this.getRecentProcessData(24),
      this.getActiveAlerts(),
      this.getRecommendations()
    ]);

    const totalReadings = processData.length;
    const avgQuality = processData.reduce((sum, reading) =>
      sum + (reading.quality_score || 0), 0) / totalReadings;
    const criticalAlerts = alerts.filter(alert => alert.severity === 'critical').length;
    const avgConfidence = recommendations.reduce((sum, rec) =>
      sum + rec.recommendations.reduce((recSum, r) => recSum + r.confidence, 0) / rec.recommendations.length, 0
    ) / recommendations.length;

    return {
      totalReadings,
      avgQuality: avgQuality || 0,
      criticalAlerts,
      avgConfidence: avgConfidence || 0,
      systemUptime: 99.2,
      energyEfficiency: 87.5,
      costSavings: 1250.30
    };
  }

}

export const dataService = new DataService();
