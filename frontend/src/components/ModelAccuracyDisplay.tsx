import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Brain, TrendingUp, BarChart3, Zap, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getModelAccuracy, ModelAccuracy } from '@/services/modelAccuracy';

export const ModelAccuracyDisplay = () => {
    const [accuracy, setAccuracy] = useState<ModelAccuracy | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

    const loadAccuracy = async () => {
        setIsLoading(true);
        try {
            const data = await getModelAccuracy();
            setAccuracy(data);
            setLastUpdated(new Date());
        } catch (error) {
            console.error('Failed to load model accuracy:', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        loadAccuracy();
        // Refresh every 30 seconds
        const interval = setInterval(loadAccuracy, 30000);
        return () => clearInterval(interval);
    }, []);

    if (isLoading || !accuracy) {
        return (
            <Card className="bg-card border-border shadow-elegant">
                <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                        <Brain className="h-5 w-5 text-primary" />
                        <span>Model Accuracy</span>
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="text-center py-8">
                        <div className="animate-spin">
                            <RefreshCw className="h-6 w-6 text-primary mx-auto" />
                        </div>
                        <p className="text-muted-foreground mt-2">Loading model accuracy...</p>
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="bg-card border-border shadow-elegant">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center space-x-2">
                        <Brain className="h-5 w-5 text-primary" />
                        <span>Model Performance Metrics</span>
                    </CardTitle>
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={loadAccuracy}
                        className="text-muted-foreground hover:text-foreground"
                    >
                        <RefreshCw className="h-4 w-4" />
                    </Button>
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                    v{accuracy.modelVersion} • {new Date(accuracy.trainedAt).toLocaleDateString()}
                </p>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Accuracy Metrics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {/* Material Classifier */}
                    <div className="bg-subtle border border-subtle rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-muted-foreground">Material Classifier</span>
                            <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                                Perfect
                            </Badge>
                        </div>
                        <div className="space-y-2">
                            <div className="text-3xl font-bold text-foreground">
                                {accuracy.materialClassifierAccuracy.toFixed(2)}%
                            </div>
                            <Progress value={accuracy.materialClassifierAccuracy} className="h-2" />
                        </div>
                        <p className="text-xs text-muted-foreground">Random Forest Algorithm</p>
                    </div>

                    {/* Quantity Regressor */}
                    <div className="bg-subtle border border-subtle rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-muted-foreground">Quantity Regressor (R²)</span>
                            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                                Excellent
                            </Badge>
                        </div>
                        <div className="space-y-2">
                            <div className="text-3xl font-bold text-foreground">
                                {accuracy.quantityRegressorR2.toFixed(2)}%
                            </div>
                            <Progress value={accuracy.quantityRegressorR2} className="h-2" />
                        </div>
                        <p className="text-xs text-muted-foreground">Gradient Boosting Algorithm</p>
                    </div>

                    {/* Quality Predictor */}
                    <div className="bg-subtle border border-subtle rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-muted-foreground">Quality Predictor (R²)</span>
                            <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200">
                                Moderate
                            </Badge>
                        </div>
                        <div className="space-y-2">
                            <div className="text-3xl font-bold text-foreground">
                                {accuracy.qualityPredictorR2.toFixed(2)}%
                            </div>
                            <Progress value={accuracy.qualityPredictorR2} className="h-2" />
                        </div>
                        <p className="text-xs text-muted-foreground">Random Forest Algorithm</p>
                    </div>
                </div>

                {/* Additional Metrics */}
                <div className="border-t border-border pt-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                            <div className="text-xs text-muted-foreground mb-1">Quantity MAE</div>
                            <div className="text-2xl font-bold text-foreground">
                                {accuracy.quantityMAE.toFixed(2)}
                            </div>
                            <div className="text-xs text-muted-foreground">kg (error)</div>
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground mb-1">Average Accuracy</div>
                            <div className="text-2xl font-bold text-foreground">
                                {accuracy.averageAccuracy.toFixed(2)}%
                            </div>
                            <div className="text-xs text-muted-foreground">overall</div>
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground mb-1">Status</div>
                            <div className="text-2xl font-bold text-green-600">✓</div>
                            <div className="text-xs text-muted-foreground">production ready</div>
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground mb-1">Last Updated</div>
                            <div className="text-xs text-foreground font-semibold">
                                {lastUpdated?.toLocaleTimeString() || 'now'}
                            </div>
                            <div className="text-xs text-muted-foreground">just now</div>
                        </div>
                    </div>
                </div>

                {/* Algorithm Details */}
                <div className="border-t border-border pt-6">
                    <h4 className="font-semibold text-foreground mb-3 flex items-center">
                        <BarChart3 className="h-4 w-4 mr-2 text-primary" />
                        Algorithm Breakdown
                    </h4>
                    <div className="space-y-2 text-sm">
                        <div className="flex items-center justify-between p-2 bg-subtle rounded">
                            <span className="text-muted-foreground">Random Forest Classifier</span>
                            <Badge variant="secondary">Material Selection</Badge>
                        </div>
                        <div className="flex items-center justify-between p-2 bg-subtle rounded">
                            <span className="text-muted-foreground">Gradient Boosting Regressor</span>
                            <Badge variant="secondary">Quantity Prediction</Badge>
                        </div>
                        <div className="flex items-center justify-between p-2 bg-subtle rounded">
                            <span className="text-muted-foreground">Random Forest Regressor</span>
                            <Badge variant="secondary">Quality Prediction</Badge>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};
