"""Feedback for small teaching tasks; unanswered prompts are never scored as passes."""
import pandas as pd


def check_first_forecast(persistence=None, historical_mean=None):
    """Illustrative closed-hour SNR values: 2, 4, 9 dB; future outcome: 5 dB."""
    if persistence is None or historical_mean is None:
        return dict(status='not_attempted', feedback='Enter both predictions before opening the worked answer.')
    if abs(float(persistence)-9)>1e-10:
        return dict(status='revise', feedback='Persistence repeats the latest available value, 9 dB; the future outcome is not an input.')
    if abs(float(historical_mean)-5)>1e-10:
        return dict(status='revise', feedback='Average the three closed input hours: (2 + 4 + 9) / 3. Do not include the future hour.')
    return dict(status='correct', feedback='Both predictions use only the history. Their absolute errors are 4 and 0 dB for this one example; this does not establish a model ranking.')


def check_label_boundary(origin, horizon_hours, boundary, proposed_label=None, proposed_train=None):
    if proposed_label is None or proposed_train is None:
        return dict(status='not_attempted', feedback='Supply the label time and whether this origin can enter training.')
    expected=pd.Timestamp(origin)+pd.Timedelta(hours=horizon_hours)
    if pd.Timestamp(proposed_label)!=expected:
        return dict(status='revise', feedback='The label belongs to origin + horizon, not to the end of the input history.')
    allowed=pd.Timestamp(origin)<pd.Timestamp(boundary) and expected<pd.Timestamp(boundary)
    if bool(proposed_train)!=allowed:
        return dict(status='revise', feedback='An early origin is insufficient: its label must arrive strictly before validation starts. A label at the boundary is purged.')
    return dict(status='correct', feedback='The input time and label arrival both respect the stated phase boundary. Explain why before moving on.')
