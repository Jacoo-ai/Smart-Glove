'''
Description: 
version: 
Author: tangshiyi
Date: 2025-03-18 16:02:28
LastEditors: tangshiyi
LastEditTime: 2025-03-27 11:49:03
'''
import random
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import Flask, request, jsonify


# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Style dictionaries
style_header = {'textAlign': 'center', 'color': '#4A90E2', 'marginBottom': '20px', 'fontSize': '36px'}
style_counter = {'textAlign': 'center', 'color': '#2C3E50', 'marginBottom': '30px', 'fontSize': '28px', 'fontWeight': 'bold'}
style_game_box = {'display': 'flex', 'justifyContent': 'space-around', 'alignItems': 'center', 'width': '80%', 'minHeight': '200px', 'position': 'relative', 'zIndex': '2'}
style_individual = {'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'width': '45%', 'padding': '15px'}
style_label = {'fontSize': '24px', 'color': '#34495E', 'textAlign': 'center', 'marginBottom': '10px', 'fontWeight': 'bold'}
style_choice = {
    'fontSize': '40px', 'color': '#000000', 'padding': '20px', 'textAlign': 'center',
    'width': '100%', 'marginBottom': '10px', 'minHeight': '80px',
    'animation': 'fadeIn 0.5s ease-in-out'
}

# Game counter
counter = 0
predicted_info = None
updated_style_choice = dict(style_choice)
display = html.Div([
            html.Div([
                html.Div("👤 Player's", style=style_label),
                html.Div("Waiting for data...", className="fade-in", style=updated_style_choice)
            ], style=style_individual),
            html.Div([
                html.Div("🤖 AI's", style=style_label),
                html.Div("Waiting for data...", className="fade-in", style=updated_style_choice)
            ], style=style_individual)
        ], style=style_game_box)

# Layout of the app
app.layout = html.Div([
    html.Video(src="./assets/bg.mp4", 
               autoPlay=True, loop=True, muted=True, 
               style={'position': 'fixed', 'top': '50%', 'left': '50%', 'transform': 'translate(-50%, -50%)', 'width': '100vw', 'height': '100vh', 'objectFit': 'cover', 'zIndex': '-1'}),
    dcc.Markdown("""
        <style>
            @keyframes fadeIn {
                from {opacity: 0; transform: scale(0.8);}
                to {opacity: 1; transform: scale(1);}
            }
            .fade-in {
                animation: fadeIn 0.5s ease-in-out;
            }
        </style>
    """, dangerously_allow_html=True),
    dcc.Store(id='predicted-class-store', storage_type='memory'),
    html.H1('Fist Dance', style=style_header),
    html.H2(id='game-counter', style=style_counter),
    html.Div(id='game-container', style=style_game_box),
    dcc.Interval(
        id='game-interval',
        interval=700,  # update every 700 ms in alternating sequence
        n_intervals=0
    )
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': 'transparent', 'height': '100vh', 'paddingTop': '50px', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexDirection': 'column'})

# label_map = {'⬆️': 0, '⬇️': 1, '⬅️': 2, '➡️': 3, '⬆️': 4, '⬇️': 5, '⬅️': 6, '➡️': 7, '⬆️': 8, '⬇️': 9, '⬅️': 10, '➡️': 11, '✊': 12, '👋': 13, '✌️': 14, '✊': 15, '👋': 16, '✌️': 17, '✊': 18, '👋': 19, '✌️': 20, '✊': 21, '👋': 22, '✌️': 23}
# reverse_dict = {v: k for k, v in label_map.items()}
simplified_label_map = {
    0: '⬆️', 4: '⬆️', 8: '⬆️',
    1: '⬇️', 5: '⬇️', 9: '⬇️',
    2: '⬅️', 6: '⬅️', 10: '⬅️',
    3: '➡️', 7: '➡️', 11: '➡️',
    12: '✊', 15: '✊', 18: '✊', 21: '✊',
    13: '👋', 16: '👋', 19: '👋', 22: '👋',
    14: '✌️', 17: '✌️', 20: '✌️', 23: '✌️'
}
winning_moves = {'✊': '👋', '👋': '✌️', '✌️': '✊'}
directions = ['⬆️', '⬇️', '⬅️', '➡️']
previous_direction = None  # Store last direction to prevent repetition

# Callback to update UI when new prediction arrives
@app.callback(
    [Output('game-container', 'children'),
     Output('game-counter', 'children')],
    [Input('game-interval', 'n_intervals')]
)
def update_game(n):
    global counter, previous_direction, predicted_info, display
    print(f"✅ UI Received: {predicted_info}")
    updated_style_choice = dict(style_choice)
    if predicted_info is None:
        display = html.Div([
            html.Div([
                html.Div("👤 Player's", style=style_label),
                html.Div("Waiting for data...", className="fade-in", style=updated_style_choice)
            ], style=style_individual),
            html.Div([
                html.Div("🤖 AI's", style=style_label),
                html.Div("Waiting for data...", className="fade-in", style=updated_style_choice)
            ], style=style_individual)
        ], style=style_game_box)
        return display, f""
    
    predicted_class = predicted_info[0]
    detected_beats = predicted_info[1]
    player_direction = simplified_label_map.get(predicted_class, "Unknown")
    if detected_beats > 4:
        if (detected_beats) % 2 == 1:  
            if detected_beats % 4 == 1: # Even intervals: Rock Paper Scissors
                if player_direction not in winning_moves:
                    player_direction = random.choice(list(winning_moves.keys()))
                ai_choice = winning_moves[player_direction]
                display = html.Div([
                    html.Div([
                        html.Div("👤 Player's", style=style_label),
                        html.Div(player_direction, className="fade-in", style=updated_style_choice)
                    ], style=style_individual),
                    html.Div([
                        html.Div("🤖 AI's", style=style_label),
                        html.Div(ai_choice, className="fade-in", style=updated_style_choice)
                    ], style=style_individual)
                ], style=style_game_box)
            else:  # Odd intervals: Direction choice
                if player_direction not in directions:
                    player_direction = random.choice(directions)
                ai_direction = player_direction  # AI mirrors player's direction
                previous_direction = player_direction  # Store last direction
                counter += 1  # Increment the counter when a full game cycle completes
                display = html.Div([
                    html.Div([
                        html.Div("👤 Player's", style=style_label),
                        html.Div(player_direction, className="fade-in", style=updated_style_choice)
                    ], style=style_individual),
                    html.Div([
                        html.Div("🤖 AI's", style=style_label),
                        html.Div(ai_direction, className="fade-in", style=updated_style_choice)
                    ], style=style_individual)
                ], style=style_game_box)
    return display, f""



@server.route('/update-prediction', methods=['POST'])
def update_prediction():
    global predicted_info
    data = request.get_json()
    predicted_info = data.get('predicted_info', None)

    if predicted_info:
        print(f"✅ Server Received: {predicted_info}")

        return jsonify({'status': 'success', 'data': predicted_info})
    else:
        return jsonify({'status': 'error', 'message': 'No prediction received'}), 400


# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
    