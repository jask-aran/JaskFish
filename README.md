# JaskFish Chess Engine

## Brief Description
JaskFish is a lightweight, interactive chess engine designed for use with a graphical user interface (GUI). It supports chess visualization and manipulation using Python, enabling users to play chess against an automated opponent or visualize moves.

## Features
- **Graphical User Interface (GUI):** Play chess through a simple, user-friendly interface built with Tkinter.
- **Engine Communication:** Send commands to and receive outputs from a chess engine running as a separate process.
- **Random Move Generation:** The engine can suggest random legal moves, ideal for testing and casual play.
- **Game Status Checks:** Detects game conditions such as checkmate, stalemate, and draw situations.
- **SVG Visualization:** Utilizes Scalable Vector Graphics (SVG) to display chess board states dynamically.
- **Support for Promotion and Move Validation:** Includes mechanisms to handle pawn promotion and validate moves.

## Installation Instructions
1. **Clone the Repository:**
   ```
   git clone https://github.com/your-github-username/JaskFish.git
   ```
2. **Install Dependencies:**
   - Ensure Python 3 is installed on your system.
   - Install required Python packages:
     ```
     pip install python-chess Pillow cairosvg
     ```

## Usage Examples
- **Starting the GUI:**
  Navigate to the project directory and run:
  ```
  python gui.py
  ```
  This command starts the chess GUI, allowing you to make moves, reset the board, or generate random moves.
  
- **Interacting with the Engine:**
  Within the GUI, use buttons like "Make Random Move" or "Reset Board" to interact with the chess engine and visualize different game scenarios.

## Configuration Options
- **Player Color Selection:**
  At the beginning of a game session, you can choose to play as White or Black, affecting the flow and strategy of the game.

## Contribution Guidelines
- **Reporting Issues:**
  Please use GitHub issues to report bugs or request features.
- **Submitting Pull Requests:**
  1. Fork the repository.
  2. Create a new branch for your changes.
  3. Implement your changes and ensure they adhere to the existing style of the codebase.
  4. Submit a pull request against the main branch.

## Testing Instructions
- **Unit Testing:**
  Run the `test.py` script within the `dev` directory to execute basic unit tests:
  ```
  python dev/test.py
  ```
- **Integration Testing:**
  Use the `test.ipynb` Jupyter notebook for more comprehensive testing and interaction with the chess engine.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements/Credits
- **Python Chess:** This project utilizes the `python-chess` library for handling chess game mechanics and move generation.
- **Tkinter:** For creating the graphical user interface.
- **CairoSVG and PIL:** For converting SVG images and managing graphical representations.

This README provides a comprehensive guide to getting started with the JaskFish Chess Engine, ensuring that users can easily set up and begin playing or testing the software.
