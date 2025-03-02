import pygame as pg
import sys
from shading import *
import pyperclip
import json
import os

def load_highlight_config():
    with open("highlights.json", "r") as file:
        return json.load(file)

class Line:
    def __init__(self, number, string, app, highlights):
        self.number = number
        self.string = string
        self.app = app
        self.highlights = highlights  # Store highlight config
        
    def add_char(self, char):
        self.string += char

    def delete_last_char(self):
        self.string = self.string[:-1]

    def render(self, offset=(0, 0), zoom=1.0):
        pg.draw.rect(self.app.screen, (18, 18, 24), (-50 - offset[0] * zoom, int((self.number * 30 - offset[1]) * zoom), 50 + len(self.string) * 16 * zoom, 30 * zoom))

        # Split string by spaces and process each token
        scaled_font = pg.font.Font("CONSOLA.TTF", int(30 * zoom))
        txt_pos_x = -offset[0] * zoom
        txt_pos_y = (self.number * 30 - offset[1]) * zoom

        inside_comment = False  # Flag to track if we're inside a comment
        i = 0
        while i < len(self.string):
            token = ''
            # Handle spaces between tokens
            if self.string[i] == ' ':
                token = ' '
                i += 1
            else:
                # Gather characters for a token
                while i < len(self.string) and self.string[i] != ' ':
                    token += self.string[i]
                    i += 1

            token_color = (255, 255, 255)  # Default white color

            # Check if this token starts a comment (and we are not already inside a comment)
            if not inside_comment and token.startswith(self.highlights["comments"]["start"]):
                inside_comment = True
                color_hex = self.highlights["comments"]["color"]
                token_color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))  # Skip '#' and convert

            # If inside a comment, render the token in the comment color (even keywords)
            if inside_comment:
                token_color = tuple(int(self.highlights["comments"]["color"][i:i+2], 16) for i in (1, 3, 5))

            # Check if this token is a keyword (when not inside a comment)
            if not inside_comment and token in self.highlights["keywords"]:
                keyword_data = self.highlights["keywords"][token]
                color_hex = keyword_data["color"]
                token_color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))  # Skip '#' and convert

            # Render the token with the selected color
            token_txt = scaled_font.render(token, True, token_color)
            self.app.screen.blit(token_txt, (txt_pos_x, txt_pos_y))
            txt_pos_x += scaled_font.size(token)[0]  # Update position for the next token

            # If we are inside a comment and reach the end of the line, stop highlighting
            if inside_comment and (token.endswith("\n") or i >= len(self.string)):
                inside_comment = False

        # Draw line number
        scaled_font = pg.font.Font("CONSOLA.TTF", int(30 * zoom))
        txt = scaled_font.render(str(self.number + 1), True, (144, 144, 144))
        xx = -offset[0] * zoom - 50
        yy = (self.number * 30 - offset[1]) * zoom
        self.app.screen.blit(txt, (xx, yy))

        return txt_pos_x, txt_pos_y  # Return position to use for cursor




class Keyboard:
    def __init__(self, app):
        self.highlights = load_highlight_config()  # Load highlight config from JSON
        self.lines = [Line(0, "", app, self.highlights)]
        self.current_line = 0
        self.app = app
        self.cursor_time = 0
        self.selection_start = None
        self.selection_end = None
        self.mouse_dragging = False  # New flag to track mouse dragging

    def add_char(self, char):
        current_line = self.lines[self.current_line]
        if self.selection_start is not None and self.selection_end is not None:
            current_line.string = (current_line.string[:self.selection_start] +
                                   char +
                                   current_line.string[self.selection_end:])
            self.clear_selection()
        else:
            current_line.add_char(char)

    def delete_char(self):
        current_line = self.lines[self.current_line]
        if self.selection_start is not None and self.selection_end is not None:
            current_line.string = (current_line.string[:self.selection_start] +
                                   current_line.string[self.selection_end:])
            self.clear_selection()
        else:
            if current_line.string:
                current_line.delete_last_char()
            else:
                if self.current_line > 0:
                    prev_line = self.lines.pop(self.current_line)
                    self.current_line -= 1
                    self.lines[self.current_line].string += prev_line.string
            self.clear_selection()

    def new_line(self):
        self.current_line += 1
        self.lines.insert(self.current_line, Line(self.current_line, "", self.app, self.highlights))
        for i, line in enumerate(self.lines):
            line.number = i
        self.clear_selection()

    def move_up(self):
        if self.current_line > 0:
            self.current_line -= 1
        self.clear_selection()

    def move_down(self):
        if self.current_line < len(self.lines) - 1:
            self.current_line += 1
        self.clear_selection()

    def toggle_selection(self, start, end):
        if self.selection_start is None:
            self.selection_start = start
            self.selection_end = end
        else:
            self.clear_selection()

    def clear_selection(self):
        self.selection_start = None
        self.selection_end = None

    def draw(self):
        for line in self.lines:
            line.render(self.app.camera.pos, self.app.camera.zoom)
        self.draw_highlight()
        self.draw_cursor()

    def draw_highlight(self):
        if self.selection_start is not None and self.selection_end is not None:
            current_line = self.lines[self.current_line]
            scaled_font = pg.font.Font("CONSOLA.TTF", int(30 * self.app.camera.zoom))
            text_widths = [scaled_font.size(current_line.string[:i])[0]
                           for i in range(len(current_line.string) + 1)]
            start_x = -self.app.camera.pos[0] * self.app.camera.zoom + text_widths[self.selection_start]
            end_index = min(self.selection_end, len(current_line.string))
            end_x = -self.app.camera.pos[0] * self.app.camera.zoom + text_widths[end_index]
            y = (self.current_line * 30 - self.app.camera.pos[1]) * self.app.camera.zoom
            height = 30 * self.app.camera.zoom
            width = end_x - start_x
            if width <= 0:
                return
            selection_rect = pg.Rect(start_x, y, width, height)
            try:
                highlighted_surface = self.app.screen.subsurface(selection_rect).copy()
                negated_surface = self.app.shader.apply_effect(highlighted_surface, "negate")
                self.app.screen.blit(negated_surface, (start_x, y))
            except Exception as e:
                print("Error applying shader effect:", e)

    def draw_cursor(self):
        scaled_font = pg.font.Font("CONSOLA.TTF", int(30 * self.app.camera.zoom))
        current_line = self.lines[self.current_line]
        text_width, _ = scaled_font.size(current_line.string)
        cursor_x = -self.app.camera.pos[0] * self.app.camera.zoom + text_width
        cursor_y = (self.current_line * 30 - self.app.camera.pos[1]) * self.app.camera.zoom
        if int(self.cursor_time * 2) % 2 == 0:
            cursor_width = 2 * self.app.camera.zoom
            cursor_height = 30 * self.app.camera.zoom
            pg.draw.rect(self.app.screen, (255, 255, 255), (cursor_x, cursor_y, cursor_width, cursor_height), 0)
        self.cursor_time += 1 / 60

    def handle_keypress(self, event):
        mods = pg.key.get_mods()
        if event.key == pg.K_BACKSPACE:
            self.delete_char()
        elif event.key == pg.K_RETURN:
            self.new_line()
        elif event.key == pg.K_UP:
            self.move_up()
        elif event.key == pg.K_DOWN:
            self.move_down()
        elif event.key == pg.K_c and mods & pg.KMOD_CTRL:
            self.copy()
        elif event.key == pg.K_v and mods & pg.KMOD_CTRL:
            self.paste()
        elif event.key == pg.K_a and mods & pg.KMOD_CTRL:
            self.select_all()
        elif event.key == pg.K_s and mods & pg.KMOD_CTRL:
            with open(self.app.source, "w") as e:
                for line in self.lines:
                    e.write(f"{line.string}\n")
            print("saved")
        elif event.unicode and event.unicode.isprintable():
            self.add_char(event.unicode)

    def copy(self):
        if self.selection_start is not None and self.selection_end is not None:
            current_line = self.lines[self.current_line]
            copied_text = current_line.string[self.selection_start:self.selection_end]
            pyperclip.copy(copied_text)

    def paste(self):
        pasted_text = pyperclip.paste()
        current_line = self.lines[self.current_line]
        if self.selection_start is not None and self.selection_end is not None:
            current_line.string = (current_line.string[:self.selection_start] +
                                   pasted_text +
                                   current_line.string[self.selection_end:])
        else:
            current_line.string += pasted_text
        self.clear_selection()

    def select_all(self):
        current_line = self.lines[self.current_line]
        self.selection_start = 0
        self.selection_end = len(current_line.string)

    def on_mouse_button_down(self, event):
        # Calculate mouse position in line coordinates
        mouse_x, mouse_y = event.pos
        mouse_x = (mouse_x + self.app.camera.pos[0]) / self.app.camera.zoom
        mouse_y = (mouse_y + self.app.camera.pos[1]) / self.app.camera.zoom
        
        line = self.lines[self.current_line]
        scaled_font = pg.font.Font("CONSOLA.TTF", int(30 * self.app.camera.zoom))
        text_widths = [scaled_font.size(line.string[:i])[0] for i in range(len(line.string) + 1)]
        
        # Determine the index of the character under the mouse
        for i, width in enumerate(text_widths):
            if mouse_x <= width:
                self.selection_start = i
                break
        self.selection_end = self.selection_start
        self.mouse_dragging = True

    def on_mouse_motion(self, event):
        if self.mouse_dragging:
            # Update selection end based on current mouse position
            mouse_x, mouse_y = event.pos
            mouse_x = (mouse_x + self.app.camera.pos[0]) / self.app.camera.zoom
            mouse_y = (mouse_y + self.app.camera.pos[1]) / self.app.camera.zoom
            
            line = self.lines[self.current_line]
            scaled_font = pg.font.Font("CONSOLA.TTF", int(30 * self.app.camera.zoom))
            text_widths = [scaled_font.size(line.string[:i])[0] for i in range(len(line.string) + 1)]
            
            for i, width in enumerate(text_widths):
                if mouse_x <= width:
                    self.selection_end = i
                    break

    def on_mouse_button_up(self, event):
        self.mouse_dragging = False



class Camera:
    def __init__(self, app):
        self.pos = self.x, self.y = 0, 0
        self.zoom = 1.0  # Current zoom level
        self.target_zoom = 1.0  # Target zoom level
        self.app = app

    def update(self):
        self.x += (pg.mouse.get_pos()[0] - self.app.width // 2 - self.x) / 12
        self.y += (pg.mouse.get_pos()[1] - self.app.height // 2 - self.y) / 12
        self.pos = (self.x, self.y)

        # Smooth zoom interpolation (lerp)
        self.zoom += (self.target_zoom - self.zoom) * 0.1  # Adjust 0.1 for speed

    def adjust_zoom(self, zoom_change):
        """Adjust target zoom within limits."""
        self.target_zoom = max(0.5, min(2.0, self.target_zoom + zoom_change))  # Clamp zoom

class Window:
    def __init__(self):
        pg.init()
        pg.mixer.init()  # Initialize the mixer for sound
        
        self.source = "example.txt"
        self.res = self.width, self.height = 800, 600
        self.screen = pg.display.set_mode(self.res)  # Borderless window
        self.clock = pg.time.Clock()
        self.keyboard = Keyboard(self)
        
        # Load the file into the editor
        with open(self.source, "r") as lines:
            for line_number, line in enumerate(lines, start=1):
                self.keyboard.lines.append(Line(line_number, line.strip(), self, load_highlight_config()))
                
        self.name = Line(-3, f"Currently Opened: {self.source}", self, load_highlight_config())
        self.name2 = Line(-2, f"Alt+S to open settings", self, load_highlight_config())
        self.shader = MangoShading()
        self.camera = Camera(self)
        self.font = pg.font.Font("CONSOLA.TTF", 30)
        self.logo = pg.image.load("logo.png")
        
        pg.display.set_icon(self.logo)
        self.orig = self.logo
        
        # --- Menu & File Navigation Attributes ---
        self.menu_active = False         # Whether the menu is visible (toggled with Alt+S)
        self.menu_page = 0               # 0: Files, 1: Settings
        self.menu_selected_index = 0     # Which option is selected in the menu
        self.current_dir = os.getcwd()   # Track the current directory
        self.dir_files = self.get_dir_files()
        self.effect_toggles = {"bloom": True, "chromatic_aberration": True}
        self.menu_offset_x = -self.width // 2  # For smooth slide-in/out
        self.volume = 1.0  # Default volume for background music
        

        # --- New File Prompt Attributes ---
        self.new_file_mode = False
        self.new_file_name = ""
        
        # Load sounds
        self.bgmusic = pg.mixer.Sound("bgmusic.mp3")
        self.open_sound = pg.mixer.Sound("open.wav")
        self.select_sound = pg.mixer.Sound("select.wav")
        
        # Play background music on loop
        self.bgmusic.play(loops=-1, maxtime=0, fade_ms=0)
        self.bgmusic.set_volume(self.volume)

    def get_dir_files(self):
        # Build the options list for the Files page:
        items = []
        parent = os.path.dirname(self.current_dir)
        if parent and parent != self.current_dir:
            items.append("..")
        try:
            contents = sorted(os.listdir(self.current_dir))
        except Exception as e:
            contents = []
        items.extend(contents)
        items.append("New File")
        return items
    
    def draw(self):
        self.screen.fill((10,10,12))
        
        # Update menu sliding animation
        target_offset = 0 if self.menu_active else -self.width // 2
        speed = 20  # Adjust for smoother/slower sliding
        if self.menu_offset_x < target_offset:
            self.menu_offset_x = min(target_offset, self.menu_offset_x + speed)
        elif self.menu_offset_x > target_offset:
            self.menu_offset_x = max(target_offset, self.menu_offset_x - speed)
        
        if self.menu_active:
            self.keyboard.draw()
            self.name.render(offset=self.camera.pos, zoom=self.camera.zoom)
            self.name2.render(offset=self.camera.pos, zoom=self.camera.zoom)
            self.draw_menu()
        else:
            self.keyboard.draw()
            self.name.render(offset=self.camera.pos, zoom=self.camera.zoom)
            self.name2.render(offset=self.camera.pos, zoom=self.camera.zoom)
        
        # If new file prompt is active, draw it over the menu
        if self.new_file_mode:
            self.draw_new_file_prompt()
            
        
        # Apply shader effects based on toggles
        effect1 = "bloom" if self.effect_toggles["bloom"] else "none"
        surface_with_effect = self.shader.apply_effect(self.screen, effect1)
        effect2 = "chromatic_aberration" if self.effect_toggles["chromatic_aberration"] else "none"
        final_surface = self.shader.apply_effect(surface_with_effect, effect2)
        self.screen.blit(final_surface, (0, 0))

        
        
        pg.display.flip()
    
    def draw_menu(self):
        # Draw a left-half menu using the sliding offset (menu_offset_x)
        menu_rect = pg.Rect(self.menu_offset_x, 0, self.width // 2, self.height)
        pg.draw.rect(self.screen, (20,20,20), menu_rect)
        
        # Header for the menu
        header_text = "Files" if self.menu_page == 0 else "Settings"
        header_render = self.font.render(header_text, True, (255,255,255))
        self.screen.blit(header_render, (self.menu_offset_x + 20, 20))
        
        # Get the options based on current page
        if self.menu_page == 0:
            options = self.dir_files
        else:
            options = ["Bloom", "Chromatic Aberration", f"Volume: {int(self.volume * 100)}%"]
        
        for i, option in enumerate(options):
            color = (0,255,0) if i == self.menu_selected_index else (255,255,255)
            if self.menu_page == 1:
                # Append toggle status for settings options
                if option == "Bloom":
                    status = "ON" if self.effect_toggles["bloom"] else "OFF"
                elif option == "Chromatic Aberration":
                    status = "ON" if self.effect_toggles["chromatic_aberration"] else "OFF"
                else:
                    # Display volume in the settings menu
                    text = f"{option}: {int(self.volume * 100)}%"
                text = option
            else:
                text = option
            option_render = self.font.render(text, True, color)
            self.screen.blit(option_render, (self.menu_offset_x + 20, 60 + i * 40))
    
    def draw_new_file_prompt(self):
        # Draw an overlay prompt for new file input
        overlay = pg.Surface((self.width, self.height))
        overlay.set_alpha(200)
        overlay.fill((0,0,0))
        self.screen.blit(overlay, (0,0))
        prompt_text = "Enter new file name: " + self.new_file_name
        prompt_render = self.font.render(prompt_text, True, (255,255,255))
        self.screen.blit(prompt_render, (self.width//2 - prompt_render.get_width()//2, self.height//2))
    
    def update(self):
        pg.display.set_caption(f"FPS: {self.clock.get_fps()} | Zoom: {self.camera.zoom:.2f}")
        self.camera.update()
        self.clock.tick()
    
    def run(self):
        while True:
            self.update()
            self.draw()
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll up
                        self.camera.adjust_zoom(0.1)
                    elif event.button == 5:  # Scroll down
                        self.camera.adjust_zoom(-0.1)
                    self.keyboard.on_mouse_button_down(event)
                elif event.type == pg.MOUSEMOTION:
                    self.keyboard.on_mouse_motion(event)
                elif event.type == pg.MOUSEBUTTONUP:
                    self.keyboard.on_mouse_button_up(event)
                elif event.type == pg.KEYDOWN:
                    mods = pg.key.get_mods()
                    # When new file prompt is active, handle its input exclusively
                    if self.new_file_mode:
                        self.handle_new_file_input(event)
                    # Toggle the menu with Alt+S
                    elif (mods & pg.KMOD_ALT) and event.key == pg.K_s:
                        self.menu_active = not self.menu_active
                        self.menu_selected_index = 0
                    elif self.menu_active:
                        self.handle_menu_navigation(event)
                    else:
                        self.keyboard.handle_keypress(event)
    
    def handle_menu_navigation(self, event):
        # Play select sound when navigating
        if event.key in [pg.K_UP, pg.K_DOWN]:
            self.select_sound.play()
        
        # Determine number of options based on the current menu page
        if self.menu_page == 0:
            num_options = len(self.dir_files)
        else:
            num_options = 3  # Three settings options now
        if event.key == pg.K_UP:
            self.menu_selected_index = (self.menu_selected_index - 1) % num_options
        elif event.key == pg.K_DOWN:
            self.menu_selected_index = (self.menu_selected_index + 1) % num_options
        elif event.key in (pg.K_LEFT, pg.K_RIGHT):
            # Switch between Files and Settings pages
            self.menu_page = 0 if self.menu_page == 1 else 1
            self.menu_selected_index = 0
        elif event.key == pg.K_RETURN:
            if self.menu_page == 0:
                selected = self.dir_files[self.menu_selected_index]
                if selected == "..":
                    # Go back to the parent directory
                    self.current_dir = os.path.dirname(self.current_dir)
                    os.chdir(self.current_dir)
                    self.dir_files = self.get_dir_files()
                    self.menu_selected_index = 0
                elif selected == "New File":
                    # Activate new file prompt
                    self.new_file_mode = True
                    self.new_file_name = ""
                else:
                    full_path = os.path.join(self.current_dir, selected)
                    if os.path.isdir(full_path):
                        self.current_dir = full_path
                        os.chdir(self.current_dir)
                        self.dir_files = self.get_dir_files()
                        self.menu_selected_index = 0
                    else:
                        # Play open sound when opening a file
                        self.open_sound.play()
                        # Load the selected file into the editor
                        self.source = full_path
                        self.keyboard.lines = [Line(0, "", self, load_highlight_config())]
                        try:
                            with open(self.source, "r") as f:
                                for line_number, line in enumerate(f, start=1):
                                    self.keyboard.lines.append(Line(line_number, line.strip(), self, load_highlight_config()))
                        except Exception as e:
                            print("Error loading file:", e)
                        self.name = Line(-3, f"Currently Opened: {self.source}", self, load_highlight_config())
                        self.menu_active = False
            elif self.menu_page == 1:
                # Toggle settings based on selection
                if self.menu_selected_index == 0:
                    self.effect_toggles["bloom"] = not self.effect_toggles["bloom"]
                elif self.menu_selected_index == 1:
                    self.effect_toggles["chromatic_aberration"] = not self.effect_toggles["chromatic_aberration"]
                elif self.menu_selected_index == 2:
                    # Adjust volume
                    if self.volume != 0.0:
                        self.volume = max(0.0, self.volume - 0.1)
                    else:
                        self.volume = 1.0
                    self.bgmusic.set_volume(self.volume)

    def handle_new_file_input(self, event):
        if event.key == pg.K_RETURN:
            # Create the new file in the current directory
            new_file_path = os.path.join(self.current_dir, self.new_file_name)
            try:
                with open(new_file_path, "w") as f:
                    f.write("")  # Create an empty file
                # Load the new file into the editor
                self.source = new_file_path
                self.keyboard.lines = [Line(0, "", self, load_highlight_config())]
                self.name = Line(-3, f"Currently Opened: {self.source}", self, load_highlight_config())
                self.menu_active = False
            except Exception as e:
                print("Error creating file:", e)
            self.new_file_mode = False
            self.dir_files = self.get_dir_files()
        elif event.key == pg.K_ESCAPE:
            # Cancel new file prompt
            self.new_file_mode = False
        elif event.key == pg.K_BACKSPACE:
            self.new_file_name = self.new_file_name[:-1]
        else:
            if event.unicode.isprintable():
                self.new_file_name += event.unicode
                    
if __name__ == "__main__":
    app = Window()
    app.run()
