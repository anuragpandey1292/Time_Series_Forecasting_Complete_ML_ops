# Forecasting Specification v1.0

## 1. Business Objective

Forecast future daily store-family sales to support downstream
demand planning and business decision-making.

## 2. Prediction Target

The prediction target is future daily sales, used as the observable
proxy for demand.

## 3. Forecast Grain

Each prediction corresponds to:

store_nbr × family × date

There are 54 stores and 33 product families, resulting in 1,782
store-family series.

## 4. Forecast Horizon

The production forecast horizon is 16 consecutive days.

Production forecast period:

2017-08-16 through 2017-08-31.

## 5. Forecast Origin

For the production forecast, the forecast origin is:

2017-08-15

Only information available at or before the forecast origin may be
used unless a feature is explicitly documented as known in advance.

## 6. Information Availability

The following information is considered available for forecasting:

- Forecast date
- Store identifier
- Product family
- Historical sales occurring before the forecast origin
- Promotion information when it is known in advance
- Store metadata
- Calendar information
- Holiday/event information when known in advance

Future feature availability must be explicitly verified before using
external variables such as transactions or oil prices.

## 7. Leakage Constraints

The model must not use:

- Actual sales from the forecast horizon
- Future target values
- Any feature value that would not have been available at the
  forecast origin

Historical sales may only be used when they occurred before the
forecasting point.

## 8. Multi-Step Forecasting

The system must support forecasting multiple consecutive future days.

The forecasting strategy must explicitly handle the information
available for each future day.

If recursive forecasting is used, predictions for earlier forecast
days may become inputs for later forecast days. Actual future sales
must not be substituted for those predictions during inference.

## 9. Validation Strategy

Model validation must simulate the production forecasting problem.

Validation should use historical forecast origins and evaluate the
following 16 days after each origin.

Each validation fold should:

1. Select a historical forecast origin.
2. Train using only information available up to that origin.
3. Generate forecasts for the following 16 days.
4. Compare predictions against the actual sales observed during
   those 16 days.
5. Repeat for multiple historical forecast origins.

This is a walk-forward / rolling-origin validation strategy.

## 10. Primary Modeling Question

The primary modeling question is:

How accurately can the forecasting system predict the next 16 days
of store-family sales using only information that would have been
available when the forecast was generated?

## 11. Feature Availability Contract

Feature availability must be explicitly documented before a feature
is included in the production model.

In particular:

- `onpromotion` must be verified as known for the forecast horizon.
- Future transactions must not be used unless their availability
  at forecast time is established.
- Future oil prices must not be used unless their availability at
  forecast time is established.
- Future holiday/event information may be used when it is known in
  advance.

## 12. Scope

This specification defines the forecasting problem and information
constraints.

It does not yet define:

- The final model algorithm
- Feature engineering implementation
- Hyperparameters
- MLflow configuration
- API design
- Deployment architecture
- Monitoring thresholds