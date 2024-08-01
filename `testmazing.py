import pygame
import numpy as np
import random
from collections import deque
import time
import heapq

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

WIDTH, HEIGHT = 1200, 800
win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Advanced Maze Solver")
clock = pygame.time.Clock()

COLORS = {
    'background': (20, 20, 30),
    'wall': (100, 100, 150),
    'player': (255, 50, 50),
    'path': (50, 255, 150),
    'visited': (100, 100, 255, 50),
    'frontier': (255, 255, 100, 100),
    'solution': (50, 200, 50),
    'button': (80, 80, 100),
    'button_hover': (100, 100, 120),
    'text': (220, 220, 220),
    'end': (0, 255, 0),
    'arrow': (255, 255, 0),
}

class Button:
    def __init__(self, x, y, width, height, text, color, text_color, function):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = COLORS['button_hover']
        self.text_color = text_color
        self.function = function

    def draw(self, surface):
        color = self.hover_color if self.is_hovered() else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        font = pygame.font.SysFont(None, 24)
        text = font.render(self.text, True, self.text_color)
        surface.blit(text, (self.rect.x + (self.rect.width - text.get_width()) // 2,
                            self.rect.y + (self.rect.height - text.get_height()) // 2))

    def is_hovered(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class Maze:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.grid = np.ones((rows, cols), dtype=int)
        self.start, self.end = (1, 1), (rows-2, cols-2)
        self.player_pos = self.start
        self.visited_cells, self.frontier_cells = set(), set()
        self.generate()

    def generate(self):
        self.grid = np.ones((self.rows, self.cols), dtype=int)
        stack = [self.start]
        visited = set([self.start])

        while stack:
            current = stack.pop()
            neighbors = self.get_unvisited_neighbors(current, visited)

            if neighbors:
                stack.append(current)
                next_cell = random.choice(neighbors)
                self.grid[next_cell] = 0
                self.grid[(current[0] + next_cell[0]) // 2, (current[1] + next_cell[1]) // 2] = 0
                visited.add(next_cell)
                stack.append(next_cell)

        self.add_loops()
        self.grid[self.start] = self.grid[self.end] = 0

    def get_unvisited_neighbors(self, pos, visited):
        r, c = pos
        return [(nr, nc) for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]
                if 0 < (nr := r + dr) < self.rows-1 and 0 < (nc := c + dc) < self.cols-1
                and (nr, nc) not in visited]

    def add_loops(self):
        for _ in range(int(self.rows * self.cols * 0.05)):
            r, c = random.randint(1, self.rows-2), random.randint(1, self.cols-2)
            if self.grid[r, c] == 1:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                open_neighbors = [n for n in neighbors if 0 <= n[0] < self.rows and 0 <= n[1] < self.cols and self.grid[n] == 0]
                if len(open_neighbors) >= 2:
                    self.grid[r, c] = 0

    def get_valid_moves(self, pos):
        r, c = pos
        return [(nr, nc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= (nr := r + dr) < self.rows and 0 <= (nc := c + dc) < self.cols
                and self.grid[nr, nc] == 0]

    def draw(self, surface, cell_size):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    pygame.draw.rect(surface, COLORS['wall'], (c * cell_size, r * cell_size, cell_size, cell_size))

def draw_cell(surface, color, x, y, cell_size, padding=1):
    pygame.draw.rect(surface, color, (x + padding, y + padding, cell_size - 2*padding, cell_size - 2*padding))

def animate_solution(surface, solution_path, progress, cell_size):
    if not solution_path:
        return
    start_color, end_color = COLORS['player'], COLORS['solution']
    for i, cell in enumerate(solution_path[:int(progress)]):
        x, y = cell[1] * cell_size, cell[0] * cell_size
        color = [start_color[j] + (end_color[j] - start_color[j]) * i / len(solution_path) for j in range(3)]
        draw_cell(surface, color, x, y, cell_size)
        if i < len(solution_path) - 1:
            next_cell = solution_path[i + 1]
            dx, dy = next_cell[1] - cell[1], next_cell[0] - cell[0]
            arrow_color = COLORS['arrow']
            center = (x + cell_size // 2, y + cell_size // 2)
            arrow_size = cell_size // 3
            if dx == 1:
                pygame.draw.polygon(surface, arrow_color, [
                    (center[0], center[1] - arrow_size),
                    (center[0] + arrow_size, center[1]),
                    (center[0], center[1] + arrow_size)
                ])
            elif dx == -1:
                pygame.draw.polygon(surface, arrow_color, [
                    (center[0], center[1] - arrow_size),
                    (center[0] - arrow_size, center[1]),
                    (center[0], center[1] + arrow_size)
                ])
            elif dy == 1:
                pygame.draw.polygon(surface, arrow_color, [
                    (center[0] - arrow_size, center[1]),
                    (center[0], center[1] + arrow_size),
                    (center[0] + arrow_size, center[1])
                ])
            elif dy == -1:
                pygame.draw.polygon(surface, arrow_color, [
                    (center[0] - arrow_size, center[1]),
                    (center[0], center[1] - arrow_size),
                    (center[0] + arrow_size, center[1])
                ])

def solve_step(maze, method, start, end):
    if method == "dfs":
        return dfs_step(maze, start, end)
    elif method == "bfs":
        return bfs_step(maze, start, end)
    elif method == "a_star":
        return a_star_step(maze, start, end)
    elif method == "dijkstra":
        return dijkstra_step(maze, start, end)
    elif method == "greedy_bfs":
        return greedy_bfs_step(maze, start, end)

def dfs_step(maze, start, end):
    if not hasattr(maze, 'dfs_stack'):
        maze.dfs_stack, maze.dfs_visited, maze.dfs_parent = [start], set(), {start: None}
    if maze.dfs_stack:
        current = maze.dfs_stack.pop()
        if current == end:
            return reconstruct_path(maze.dfs_parent, end), True
        if current not in maze.dfs_visited:
            maze.dfs_visited.add(current)
            maze.visited_cells.add(current)
            for neighbor in maze.get_valid_moves(current):
                if neighbor not in maze.dfs_visited:
                    maze.dfs_parent[neighbor] = current
                    maze.dfs_stack.append(neighbor)
                    maze.frontier_cells.add(neighbor)
        return [current], False
    return None, True

def bfs_step(maze, start, end):
    if not hasattr(maze, 'bfs_queue'):
        maze.bfs_queue, maze.bfs_visited, maze.bfs_parent = deque([start]), set(), {start: None}
    if maze.bfs_queue:
        current = maze.bfs_queue.popleft()
        if current == end:
            return reconstruct_path(maze.bfs_parent, end), True
        if current not in maze.bfs_visited:
            maze.bfs_visited.add(current)
            maze.visited_cells.add(current)
            for neighbor in maze.get_valid_moves(current):
                if neighbor not in maze.bfs_visited:
                    maze.bfs_parent[neighbor] = current
                    maze.bfs_queue.append(neighbor)
                    maze.frontier_cells.add(neighbor)
        return [current], False
    return None, True

def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def a_star_step(maze, start, end):
    if not hasattr(maze, 'a_star_open_set'):
        maze.a_star_open_set = [(0, start)]
        maze.a_star_came_from, maze.a_star_g_score, maze.a_star_f_score = {}, {start: 0}, {start: heuristic(start, end)}
    if maze.a_star_open_set:
        current = min(maze.a_star_open_set, key=lambda x: x[0])[1]
        maze.a_star_open_set = [x for x in maze.a_star_open_set if x[1] != current]
        maze.visited_cells.add(current)
        if current == end:
            return reconstruct_path(maze.a_star_came_from, end), True
        for neighbor in maze.get_valid_moves(current):
            tentative_g_score = maze.a_star_g_score[current] + 1
            if tentative_g_score < maze.a_star_g_score.get(neighbor, float('inf')):
                maze.a_star_came_from[neighbor] = current
                maze.a_star_g_score[neighbor] = tentative_g_score
                f_score = maze.a_star_g_score[neighbor] + heuristic(neighbor, end)
                maze.a_star_f_score[neighbor] = f_score
                maze.a_star_open_set.append((f_score, neighbor))
                maze.frontier_cells.add(neighbor)
        return [current], False
    return None, True

def dijkstra_step(maze, start, end):
    if not hasattr(maze, 'dijkstra_queue'):
        maze.dijkstra_queue = [(0, start)]
        maze.dijkstra_distances = {start: 0}
        maze.dijkstra_parent = {start: None}
    if maze.dijkstra_queue:
        current_dist, current = heapq.heappop(maze.dijkstra_queue)
        maze.visited_cells.add(current)
        if current == end:
            return reconstruct_path(maze.dijkstra_parent, end), True
        for neighbor in maze.get_valid_moves(current):
            distance = current_dist + 1
            if distance < maze.dijkstra_distances.get(neighbor, float('inf')):
                maze.dijkstra_distances[neighbor] = distance
                maze.dijkstra_parent[neighbor] = current
                heapq.heappush(maze.dijkstra_queue, (distance, neighbor))
                maze.frontier_cells.add(neighbor)
        return [current], False
    return None, True

def greedy_bfs_step(maze, start, end):
    if not hasattr(maze, 'greedy_queue'):
        maze.greedy_queue = [(heuristic(start, end), start)]
        maze.greedy_visited = set()
        maze.greedy_parent = {start: None}
    if maze.greedy_queue:
        _, current = heapq.heappop(maze.greedy_queue)
        if current == end:
            return reconstruct_path(maze.greedy_parent, end), True
        if current not in maze.greedy_visited:
            maze.greedy_visited.add(current)
            maze.visited_cells.add(current)
            for neighbor in maze.get_valid_moves(current):
                if neighbor not in maze.greedy_visited:
                    maze.greedy_parent[neighbor] = current
                    heapq.heappush(maze.greedy_queue, (heuristic(neighbor, end), neighbor))
                    maze.frontier_cells.add(neighbor)
        return [current], False
    return None, True

def reconstruct_path(parent, end):
    path, current = [], end
    while current:
        path.append(current)
        current = parent.get(current)
    return path[::-1]

def generate_tone(frequency, duration=100, volume=0.1):
    sample_rate = 44100
    t = np.linspace(0, duration / 1000, int(duration * sample_rate / 1000), False)
    wave = np.sin(2 * np.pi * frequency * t) * volume
    envelope = np.linspace(0, 1, len(wave))
    envelope = np.minimum(envelope, np.linspace(1, 0, len(wave)))
    wave *= envelope
    stereo_wave = np.column_stack((wave, wave))  # Create a stereo array
    sound = pygame.sndarray.make_sound((stereo_wave * 32767).astype(np.int16))
    return sound

FREQUENCIES = [
    130.81,  # C3
    146.83,  # D3
    165.00,  # E3
    196.00,  # G3
    220.00,  # A3
    261.63,  # C4 (Middle C)
    329.63,  # E4
    392.00,  # G4
    440.00,  # A4
    523.25,  # C5
    587.33,  # D5
    659.25,  # E5
    783.99,  # G5
    880.00   # A5
]

tones = [generate_tone(freq, duration=150, volume=0.2) for freq in FREQUENCIES]
kick = generate_tone(55, duration=100, volume=0.3)  # A1 for kick
snare = generate_tone(220, duration=80, volume=0.2)  # A3 for snare

def play_step_sound(step_count):
    tones[step_count % len(tones)].play()
    if step_count % 4 == 0:
        kick.play()
    elif step_count % 2 == 1:
        snare.play()

def main():
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = 1200, 800
    win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Advanced Maze Solver")

    rows, cols = 31, 31
    cell_size = min(700 // max(rows, cols), 40)
    maze = Maze(rows, cols)
    
    solving, solve_method, solve_speed, paused = False, None, 30, False
    solution_path, steps_taken, fastest_path_length = None, 0, 0
    animation_progress, solving_complete, clearing_progress, solve_time = 0, False, 0, 0
    audio_enabled, volume = True, 0.5
    pygame.mixer.set_num_channels(8) # Increase number of audio channels

    buttons = [
        Button(WIDTH - 420, 50, 200, 40, "Generate New", COLORS['button'], COLORS['text'], lambda: Maze(rows, cols)),
        Button(WIDTH - 420, 100, 200, 40, "Reset", COLORS['button'], COLORS['text'], None),
        Button(WIDTH - 420, 150, 200, 40, "DFS Solve", COLORS['button'], COLORS['text'], lambda: "dfs"),
        Button(WIDTH - 420, 200, 200, 40, "BFS Solve", COLORS['button'], COLORS['text'], lambda: "bfs"),
        Button(WIDTH - 420, 250, 200, 40, "A* Solve", COLORS['button'], COLORS['text'], lambda: "a_star"),
        Button(WIDTH - 420, 300, 200, 40, "Dijkstra Solve", COLORS['button'], COLORS['text'], lambda: "dijkstra"),
        Button(WIDTH - 420, 350, 200, 40, "Greedy BFS Solve", COLORS['button'], COLORS['text'], lambda: "greedy_bfs"),
        Button(WIDTH - 220, 50, 200, 40, "Play/Pause", COLORS['button'], COLORS['text'], None),
        Button(WIDTH - 220, 100, 200, 40, "Step Forward", COLORS['button'], COLORS['text'], None),
        Button(WIDTH - 220, 150, 200, 40, "Increase Speed", COLORS['button'], COLORS['text'], None),
        Button(WIDTH - 220, 200, 200, 40, "Decrease Speed", COLORS['button'], COLORS['text'], None),
        Button(WIDTH - 220, 250, 95, 40, "Rows +", COLORS['button'], COLORS['text'], lambda: change_size(1, 0)),
        Button(WIDTH - 115, 250, 95, 40, "Rows -", COLORS['button'], COLORS['text'], lambda: change_size(-1, 0)),
        Button(WIDTH - 220, 300, 95, 40, "Cols +", COLORS['button'], COLORS['text'], lambda: change_size(0, 1)),
        Button(WIDTH - 115, 300, 95, 40, "Cols -", COLORS['button'], COLORS['text'], lambda: change_size(0, -1)),
        Button(WIDTH - 220, 350, 200, 40, "Toggle Audio", COLORS['button'], COLORS['text'], None),
        Button(WIDTH - 220, 400, 95, 40, "Volume +", COLORS['button'], COLORS['text'], None),
        Button(WIDTH - 115, 400, 95, 40, "Volume -", COLORS['button'], COLORS['text'], None),
    ]

    def reset_maze():
        nonlocal solving, solution_path, steps_taken, fastest_path_length, animation_progress, solving_complete, clearing_progress, solve_time
        maze.player_pos = maze.start
        maze.visited_cells.clear()
        maze.frontier_cells.clear()
        solving, solution_path, steps_taken, fastest_path_length = False, None, 0, 0
        animation_progress, solving_complete, clearing_progress, solve_time = 0, False, 0, 0

    def resize_window(width, height):
        global WIDTH, HEIGHT
        WIDTH, HEIGHT = width, height
        return pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    def change_size(row_change, col_change):
        nonlocal rows, cols, cell_size, maze
        rows, cols = max(5, min(rows + row_change, 1001)), max(5, min(cols + col_change, 1001))
        cell_size = max(min(700 // max(rows, cols), 40), 1)
        maze = Maze(rows, cols)
        reset_maze()

    maze_surface = pygame.Surface((800, 800))
    maze_surface.set_colorkey(COLORS['background'])

    while True:
        win.fill(COLORS['background'])
        maze_surface.fill(COLORS['background'])
        
        maze.draw(maze_surface, cell_size)
        
        if solving and not solving_complete:
            for cell in maze.visited_cells:
                draw_cell(maze_surface, COLORS['visited'], cell[1] * cell_size, cell[0] * cell_size, cell_size)
            for cell in maze.frontier_cells:
                draw_cell(maze_surface, COLORS['frontier'], cell[1] * cell_size, cell[0] * cell_size, cell_size)
        elif solving_complete:
            if clearing_progress < 1:
                for cell in list(maze.visited_cells) + list(maze.frontier_cells):
                    clear_color = tuple(int(COLORS['background'][i] * clearing_progress + COLORS['visited'][i] * (1 - clearing_progress)) for i in range(3))
                    draw_cell(maze_surface, clear_color, cell[1] * cell_size, cell[0] * cell_size, cell_size)
            else:
                animate_solution(maze_surface, solution_path, animation_progress, cell_size)

        draw_cell(maze_surface, COLORS['end'], maze.end[1] * cell_size, maze.end[0] * cell_size, cell_size)
        draw_cell(maze_surface, COLORS['player'], maze.player_pos[1] * cell_size, maze.player_pos[0] * cell_size, cell_size)

        win.blit(maze_surface, (0, 0))
        
        for button in buttons:
            button.draw(win)

        font = pygame.font.SysFont(None, 24)
        stats = [
            f"Steps taken: {steps_taken}",
            f"Fastest path: {fastest_path_length}",
            f"Speed: {solve_speed}",
            f"Solve time: {solve_time:.2f}s",
            f"Maze size: {rows}x{cols}",
            f"Audio: {'On' if audio_enabled else 'Off'}",
            f"Volume: {int(volume * 100)}%"
        ]
        for i, stat in enumerate(stats):
            text = font.render(stat, True, COLORS['text'])
            win.blit(text, (WIDTH - 420, 450 + i * 30))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.VIDEORESIZE:
                win = resize_window(event.w, event.h)
            if event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons:
                    if button.is_clicked(event.pos):
                        if button.text == "Generate New":
                            maze = button.function()
                            reset_maze()
                        elif button.text == "Reset":
                            reset_maze()
                        elif button.text in ["DFS Solve", "BFS Solve", "A* Solve", "Dijkstra Solve", "Greedy BFS Solve"]:
                            solve_method = button.function()
                            solving, paused, solution_path, steps_taken = True, False, None, 0
                            maze.visited_cells.clear()
                            maze.frontier_cells.clear()
                            animation_progress, solving_complete, clearing_progress = 0, False, 0
                            solve_time = time.time()
                        elif button.text == "Play/Pause":
                            paused = not paused
                        elif button.text == "Step Forward" and solution_path and int(animation_progress) < len(solution_path) - 1:
                            animation_progress = min(animation_progress + 1, len(solution_path) - 1)
                            maze.player_pos = solution_path[int(animation_progress)]
                        elif button.text == "Increase Speed":
                            solve_speed = min(solve_speed + 10, 120)
                        elif button.text == "Decrease Speed":
                            solve_speed = max(solve_speed - 10, 10)
                        elif button.text in ["Rows +", "Rows -", "Cols +", "Cols -"]:
                            button.function()
                        elif button.text == "Toggle Audio":
                            audio_enabled = not audio_enabled
                        elif button.text == "Volume +":
                            volume = min(1.0, volume + 0.1)
                        elif button.text == "Volume -":
                            volume = max(0.0, volume - 0.1)
            elif event.type == pygame.KEYDOWN and not solving:
                new_pos = maze.player_pos
                if event.key == pygame.K_UP:
                    new_pos = (maze.player_pos[0] - 1, maze.player_pos[1])
                elif event.key == pygame.K_RIGHT:
                    new_pos = (maze.player_pos[0], maze.player_pos[1] + 1)
                elif event.key == pygame.K_DOWN:
                    new_pos = (maze.player_pos[0] + 1, maze.player_pos[1])
                elif event.key == pygame.K_LEFT:
                    new_pos = (maze.player_pos[0], maze.player_pos[1] - 1)
                if new_pos in maze.get_valid_moves(maze.player_pos):
                    maze.player_pos = new_pos

        if solving and not paused:
            if not solving_complete:
                current_path, finished = solve_step(maze, solve_method, maze.player_pos, maze.end)
                steps_taken += 1
                if finished:
                    solution_path = current_path
                    fastest_path_length = len(solution_path) - 1
                    solving_complete, clearing_progress, animation_progress = True, 0, 0
                    solve_time = time.time() - solve_time
                else:
                    maze.player_pos = current_path[-1]
                
                if audio_enabled:
                    play_step_sound(steps_taken)
                    for sound in tones + [kick, snare]:
                        sound.set_volume(volume)
            elif clearing_progress < 1:
                clearing_progress += 0.05
            elif animation_progress < len(solution_path):
                animation_progress += solve_speed / 60
                maze.player_pos = solution_path[min(int(animation_progress), len(solution_path) - 1)]
                
                if audio_enabled:
                    play_step_sound(int(animation_progress))
                    for sound in tones + [kick, snare]:
                        sound.set_volume(volume)

        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()
