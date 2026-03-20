# Data Dictionary

## Shared Match Metadata (both event types)
| Field             | Type        | Values / Notes                       |
|-------------------|-------------|--------------------------------------|
| match_id          | string      | e.g. 2023_AIF_LIM_KK                 |
| season            | integer     | 2021 to 2025                         |
| competition_stage | categorical | QF / SF / Final                      |
| match_date        | date        | YYYY-MM-DD                           |
| team_a            | string      | Standardised county name             |
| team_b            | string      | Standardised county name             |
| venue             | string      | Full venue name                      |

## Puckout Fields
| Field             | Type        | Values / Notes                       |
|-------------------|-------------|--------------------------------------|
| pucking_team      | string      | County name                          |
| type              | categorical | short / medium / long                |
| delivery          | categorical | C / P / T / B / X                    |
| target_zone       | categorical | AL/AC/AR/ML/MC/MR/DL/DC/DR           |
| retained          | categorical | yes / no / unknown                   |
| next_action       | categorical | shot / carry / turnover / foul       |


## Frees & 65s Fields
| Field             | Type        | Values / Notes                       |
|-------------------|-------------|--------------------------------------|
| taking_team       | string      | County name                          |
| free_type         | categorical | Free / 65 / Sideline                 |
| shooter           | string      | Player name (optional)               |
| zone              | categorical | AL/AC/AR/ML/MC/MR/DL/DC/DR           |
| distance          | categorical | Close / Mid / Long                   |
| shot_type         | categorical | Direct / Tapped / Passed             |
| side              | categorical | Left / Central / Right               |
| under_pressure    | categorical | yes / no / unknown                   |
| result            | categorical | Point/Goal/Wide/Short/Blocked/Saved  |
