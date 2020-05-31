import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Label('Neighborhood'),
    dcc.Dropdown(
        options=[
            {'label': 'Bloomington Heights', 'value': 'Blmngtn'},
            {'label': 'Bluestem', 'value': 'Blueste'},
            {'label': 'Briardale', 'value': 'BrDale'},
            {'label': 'Brookside', 'value': 'BrkSide'},
            {'label': 'Clear Creek', 'value': 'ClearCr'},
            {'label': 'College Creek', 'value': 'CollgCr'},
            {'label': 'Crawford', 'value': 'Crawfor'},
            {'label': 'Edwards', 'value': 'Edwards'},
            {'label': 'Gilbert', 'value': 'Gilbert'},
            {'label': 'Iowa DOT and rail road', 'value': 'IDOTRR'},
            {'label': 'Meadow Village', 'value': 'MeadowV'},
            {'label': 'Mitchell', 'value': 'Mitchel'},
            {'label': 'North Ames', 'value': 'Names'},
            {'label': 'Northridge', 'value': 'NoRidge'},
            {'label': 'Northpark Villa', 'value': 'NPkVill'},
            {'label': 'Northridge Heights', 'value': 'NridgHt'},
            {'label': 'Northwest Ames', 'value': 'NWAmes'},
            {'label': 'Old town', 'value': 'OldTown'},
            {'label': 'South West of Iowa State University', 'value': 'SWISU'},
            {'label': 'Sawyer', 'value': 'Sawyer'},
            {'label': 'Sawyer West', 'value': 'SawyerW'},
            {'label': 'Somerset', 'value': 'Somerst'},
            {'label': 'Stone Brook', 'value': 'StoneBr'},
            {'label': 'Timberland', 'value': 'Timber'},
            {'label': 'Veenker', 'value': 'Veenker'}
        ],
        value=''
    ),

    html.Label('Sub class'),
    dcc.Dropdown(
        options=[
            {'label': '1-STORY 1946 & NEWER ALL STYLES', 'value': '20'},
            {'label': '1-STORY 1945 & OLDER', 'value': '30'},
            {'label': '1-STORY W/FINISHED ATTIC ALL AGES', 'value': '40'},
            {'label': '1-1/2 STORY - UNFINISHED ALL AGES', 'value': '45'},
            {'label': '1-1/2 STORY FINISHED ALL AGES', 'value': '50'},
            {'label': '2-STORY 1946 & NEWER', 'value': '60'},
            {'label': '2-STORY 1945 & OLDER', 'value': '70'},
            {'label': '2-1/2 STORY ALL AGES', 'value': '75'},
            {'label': 'SPLIT OR MULTI-LEVEL', 'value': '80'},
            {'label': 'SPLIT FOYER', 'value': '85'},
            {'label': 'DUPLEX - ALL STYLES AND AGES', 'value': '90'},
            {'label': '1-STORY PUD (Planned Unit Development) - 1946 & NEWER', 'value': '120'},
            {'label': '1-1/2 STORY PUD - ALL AGES', 'value': '150'},
            {'label': '2-STORY PUD - 1946 & NEWER', 'value': '160'},
            {'label': 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER', 'value': '180'},
            {'label': '2 FAMILY CONVERSION - ALL STYLES AND AGES', 'value': '190'}
        ],
        value=''
    ),

    html.Label('Zone'),
    dcc.Dropdown(
        options=[
            {'label': 'Agriculture', 'value': 'A'},
            {'label': 'Commercial', 'value': 'C'},
            {'label': 'Floating Village Residential', 'value': 'FV'},
            {'label': 'Industrial', 'value': 'I'},
            {'label': 'Residential High Density', 'value': 'RH'},
            {'label': 'Residential Low Density', 'value': 'RL'},
            {'label': 'Residential Low Density Park', 'value': 'RP'},
            {'label': 'Residential Medium Density', 'value': 'RM'}
        ],
        value=''
    ),

    html.Label('Lot frontage'),
    dcc.Input(
        placeholder='Enter a value...',
        type='number',
        value=''
    ),

    html.Label('Lot area'),
    dcc.Input(
        placeholder='Enter a value...',
        type='number',
        value=''
    ),

    html.Label('Street'),
    dcc.RadioItems(
        options=[
            {'label': 'Gravel', 'value': 'Grvl'},
            {'label': 'Paved', 'value': 'Pave'}
        ],
        value=''
    ),

    html.Label('Lot shape'),
    dcc.RadioItems(
        options=[
            {'label': 'Regular', 'value': 'Reg'},
            {'label': 'Slightly irregular', 'value': 'IR1'},
            {'label': 'Moderately Irregular', 'value': 'IR2'},
            {'label': 'Irregular', 'value': 'IR3'}
        ],
        value=''
    ),

    html.Label('Land contour'),
    dcc.RadioItems(
        options=[
            {'label': 'Near Flat', 'value': 'Lvl'},
            {'label': 'Banked', 'value': 'Bnk'},
            {'label': 'Hillside', 'value': 'HLS'},
            {'label': 'Depression', 'value': 'Low'}
        ],
        value=''
    ),

    html.Label('Utilities'),
    dcc.RadioItems(
        options=[
            {'label': 'All public Utilities', 'value': 'AllPub'},
            {'label': 'Electricity, Gas, and Water', 'value': 'NoSewr'},
            {'label': 'Electricity and Gas Only', 'value': 'NoSeWa'},
            {'label': 'Electricity only', 'value': 'ELO'}
        ],
        value=''
    ),

    html.Label('Lot configuration'),
    dcc.RadioItems(
        options=[
            {'label': 'Inside lot', 'value': 'Inside'},
            {'label': 'Corner lot', 'value': 'Corner'},
            {'label': 'Cul-de-Sac', 'value': 'CulDSac'},
            {'label': 'Frontage on 2 sides of property', 'value': 'FR2'},
            {'label': 'Frontage on 3 sides of property', 'value': 'FR3'}
        ],
        value=''
    ),

    html.Label('Landslope'),
    dcc.RadioItems(
        options=[
            {'label': 'Gentle slope', 'value': 'Gtl'},
            {'label': 'Moderate slope', 'value': 'Mod'},
            {'label': 'Severe slope', 'value': 'Sev'}
        ],
        value=''
    ),

    html.Label('Condition'),
    dcc.Dropdown(
        options=[
            {'label': 'Adjacent to arterial street', 'value': 'Artery'},
            {'label': 'Adjacent to feeder street', 'value': 'Feedr'},
            {'label': 'Normal', 'value': 'Norm'},
            {'label': 'Within 200\' of North-South Railroad', 'value': 'RRNn'},
            {'label': 'Adjacent to North-South Railroad', 'value': 'RRAn'},
            {'label': 'Near positive off-site feature--park, greenbelt, etc.', 'value': 'PosN'},
            {'label': 'Adjacent to postive off-site feature', 'value': 'PosA'},
            {'label': 'Within 200\' of East-West Railroad', 'value': 'RRNe'},
            {'label': 'Adjacent to East-West Railroad', 'value': 'RRAe'}
        ],
        value=''
    ),

    html.Label('If more than one'),
    dcc.Dropdown(
        options=[
            {'label': 'Adjacent to arterial street', 'value': 'Artery'},
            {'label': 'Adjacent to feeder street', 'value': 'Feedr'},
            {'label': 'Normal', 'value': 'Norm'},
            {'label': 'Within 200\' of North-South Railroad', 'value': 'RRNn'},
            {'label': 'Adjacent to North-South Railroad', 'value': 'RRAn'},
            {'label': 'Near positive off-site feature--park, greenbelt, etc.', 'value': 'PosN'},
            {'label': 'Adjacent to postive off-site feature', 'value': 'PosA'},
            {'label': 'Within 200\' of East-West Railroad', 'value': 'RRNe'},
            {'label': 'Adjacent to East-West Railroad', 'value': 'RRAe'}
        ],
        value=''
    ),

    # html.Label('Type of dwelling'),
    # dcc.Dropdown

], style={'columnCount': 1})

if __name__ == '__main__':
    app.run_server(debug=True)