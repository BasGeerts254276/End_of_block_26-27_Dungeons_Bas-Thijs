"""PyGame view. Deliberately thin: the sim runs fine without this file."""
from __future__ import annotations

import pygame

from engine import GameState
from world import CHEST, ENTRY, WALL

TILE = 22
SIDEBAR = 320
FOG = (26, 26, 34)
UNSEEN = (12, 12, 16)
WALL_C = (70, 72, 88)
FLOOR_C = (140, 142, 158)
WALL_DIM = (40, 42, 54)
FLOOR_DIM = (72, 74, 88)
CHEST_C = (214, 170, 62)
SMOKE_C = (150, 255, 40)         # toxic neon green, stands out against grey
SMOKE_EDGE = (215, 255, 150)
ENTRY_C = (90, 130, 120)
TEXT = (226, 226, 234)
DIM_TEXT = (150, 150, 165)

ROLE_COLORS = {
    "Tank": (86, 148, 214),
    "DPS": (214, 96, 96),
    "Healer": (120, 200, 130),
    "Scout": (206, 170, 220),
}
MONSTER_C = (188, 84, 130)


class Renderer:
    def __init__(self, state: GameState, fps: float = 8,
                 paused: bool = False):
        pygame.display.init()   # no mixer: avoids ALSA noise
        pygame.font.init()
        self.state = state
        self.fps = fps
        w = state.gmap.width * TILE + SIDEBAR
        h = max(state.gmap.height * TILE, 560)
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Warband — Agentic Dungeon")
        self.font = pygame.font.SysFont("monospace", 13)
        self.big = pygame.font.SysFont("monospace", 20, bold=True)
        self.clock = pygame.time.Clock()
        self.paused = paused
        self._step = False

    def pump(self) -> bool:
        """False when the user wants to quit.

        Space pauses and resumes. N (or the right arrow) plays exactly one
        turn while paused.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key in (pygame.K_n, pygame.K_RIGHT):
                    self._step = True
        return True

    def should_advance(self) -> bool:
        """True when the sim may play the next turn."""
        if not self.paused:
            return True
        if self._step:
            self._step = False
            return True
        return False

    def draw(self) -> None:
        s = self.state
        self.screen.fill(UNSEEN)

        for y in range(s.gmap.height):
            for x in range(s.gmap.width):
                p = (x, y)
                rect = pygame.Rect(x * TILE, y * TILE, TILE - 1, TILE - 1)
                if p not in s.known:
                    continue
                lit = p in s.visible
                ch = s.gmap.char(p)
                if ch == WALL:
                    color = WALL_C if lit else WALL_DIM
                elif ch == CHEST:
                    color = CHEST_C if lit else (110, 90, 40)
                elif ch == ENTRY:
                    color = ENTRY_C
                else:
                    color = FLOOR_C if lit else FLOOR_DIM
                pygame.draw.rect(self.screen, color, rect)
                if not lit:
                    overlay = pygame.Surface((TILE - 1, TILE - 1))
                    overlay.set_alpha(90)
                    overlay.fill(FOG)
                    self.screen.blit(overlay, rect.topleft)

        for c in s.smokes:
            # Thick while at full size, see-through while it collapses.
            alpha = 120 if c.fading else 210
            for p in c.tiles:
                if p not in s.known:
                    continue
                rect = pygame.Rect(p[0] * TILE, p[1] * TILE, TILE - 1, TILE - 1)
                cloud = pygame.Surface((TILE - 1, TILE - 1))
                cloud.set_alpha(alpha)
                cloud.fill(SMOKE_C)
                self.screen.blit(cloud, rect.topleft)
                pygame.draw.rect(self.screen, SMOKE_EDGE, rect, 1)
                puff = self.font.render("~", True, SMOKE_EDGE)
                self.screen.blit(puff, puff.get_rect(center=rect.center))

        for m in s.living_monsters:
            if m.pos in s.visible:
                self._entity(m.pos, MONSTER_C, "M", m.hp / m.max_hp)
        for h in s.living_heroes:
            self._entity(h.pos, ROLE_COLORS.get(h.role, TEXT),
                         h.role[0], h.hp / h.max_hp)

        self._sidebar()
        pygame.display.flip()
        # While paused, keep the window responsive to key presses.
        self.clock.tick(30 if self.paused else self.fps)

    def _entity(self, pos, color, letter, hp_frac) -> None:
        x, y = pos
        cx, cy = x * TILE + TILE // 2, y * TILE + TILE // 2
        pygame.draw.circle(self.screen, color, (cx, cy), TILE // 2 - 2)
        glyph = self.font.render(letter, True, (15, 15, 20))
        self.screen.blit(glyph, glyph.get_rect(center=(cx, cy)))
        bar = pygame.Rect(x * TILE + 2, y * TILE + TILE - 4, TILE - 5, 2)
        pygame.draw.rect(self.screen, (60, 20, 25), bar)
        bar.width = int((TILE - 5) * max(0.0, hp_frac))
        pygame.draw.rect(self.screen, (90, 210, 110), bar)

    def _sidebar(self) -> None:
        s = self.state
        x0 = s.gmap.width * TILE + 12
        y = 10
        pygame.draw.rect(
            self.screen, (18, 18, 24),
            pygame.Rect(x0 - 12, 0, SIDEBAR, self.screen.get_height()))

        title = self.big.render(
            f"Turn {s.turn}/{s.max_turns}"
            f"{'   PAUSED' if self.paused else ''}", True, TEXT)
        self.screen.blit(title, (x0, y))
        y += 26
        self.screen.blit(self.font.render(
            "space = pause/play   n = next turn", True, DIM_TEXT), (x0, y))
        y += 22

        chests = s.total_chests - len(s.chests)
        self.screen.blit(self.font.render(
            f"Chests {chests}/{s.total_chests}   "
            f"Enemies {len(s.living_monsters)}", True, DIM_TEXT), (x0, y))
        y += 18
        smoke_txt = ("Smoke: " + ", ".join(
            f"{c.center} {c.turns_left}t{' thin' if c.fading else ''}"
            for c in s.smokes)
            if s.smokes else "Smoke: none")
        self.screen.blit(self.font.render(smoke_txt[:44], True, SMOKE_C),
                         (x0, y))
        y += 22

        for h in s.heroes:
            color = ROLE_COLORS.get(h.role, TEXT) if h.alive else (90, 70, 70)
            goal = s.goals.get(h.name, "-")
            label = (f"{h.name:<6}{h.role:<7}{h.hp:>3}/{h.max_hp:<3}"
                     if h.alive else f"{h.name:<6}{h.role:<7} down")
            self.screen.blit(self.font.render(label, True, color), (x0, y))
            y += 16
            if h.alive:
                self.screen.blit(self.font.render(
                    f"      goal: {goal}", True, DIM_TEXT), (x0, y))
                y += 18

        y += 4
        for m in s.living_monsters:
            if m.pos in s.visible:
                self.screen.blit(self.font.render(
                    f"{m.name:<10}{m.hp:>3}/{m.max_hp:<3} {m.order}",
                    True, MONSTER_C), (x0, y))
                y += 15

        y += 8
        pygame.draw.line(self.screen, (50, 50, 62), (x0, y),
                         (x0 + SIDEBAR - 24, y))
        y += 10

        avail = (self.screen.get_height() - y) // 15
        for line in s.log[-avail:]:
            self.screen.blit(
                self.font.render(line[:44], True, DIM_TEXT), (x0, y))
            y += 15

    def banner(self, text: str) -> None:
        surf = self.big.render(text, True, (250, 240, 200))
        rect = surf.get_rect(center=(self.state.gmap.width * TILE // 2,
                                     self.state.gmap.height * TILE // 2))
        box = rect.inflate(30, 20)
        pygame.draw.rect(self.screen, (20, 20, 28), box)
        pygame.draw.rect(self.screen, (120, 120, 140), box, 2)
        self.screen.blit(surf, rect)
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
