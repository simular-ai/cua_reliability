"""Environment perturbation definitions for reliability experiments.

Each perturbation set modifies the VM's visual/system environment via gsettings
or system commands, applied before each task via the existing "execute" setup type.

Perturbation set 1: bigger cursor, bottom dock, Adwaita icons, Asia/Tokyo timezone
Perturbation set 2: smaller cursor, right dock, Humanity icons, Europe/London timezone
"""

PERTURBATION_SETS = {
    "1": [
        {
            "name": "wallpaper",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": (
                            "gsettings set org.gnome.desktop.background picture-uri "
                            "'file:///usr/share/backgrounds/warty-final-ubuntu.png'"
                            " && gsettings set org.gnome.desktop.background picture-uri-dark "
                            "'file:///usr/share/backgrounds/warty-final-ubuntu.png'"
                        ),
                        "shell": True,
                    },
                }
            ],
        },
        {
            "name": "cursor_size",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": "gsettings set org.gnome.desktop.interface cursor-size 48",
                        "shell": True,
                    },
                }
            ],
        },
        {
            "name": "dock_position",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": "gsettings set org.gnome.shell.extensions.dash-to-dock dock-position 'BOTTOM'",
                        "shell": True,
                    },
                }
            ],
        },
        {
            "name": "icon_theme",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": "gsettings set org.gnome.desktop.interface icon-theme 'Adwaita'",
                        "shell": True,
                    },
                }
            ],
        },
        {
            "name": "timezone",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": "echo '{CLIENT_PASSWORD}' | sudo -S timedatectl set-timezone Asia/Tokyo",
                        "shell": True,
                    },
                }
            ],
        },
    ],
    "2": [
        {
            "name": "wallpaper",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": (
                            "gsettings set org.gnome.desktop.background picture-uri "
                            "'file:///usr/share/backgrounds/Optical_Fibers_in_Dark_by_Elena_Ostasheva.jpg'"
                            " && gsettings set org.gnome.desktop.background picture-uri-dark "
                            "'file:///usr/share/backgrounds/Optical_Fibers_in_Dark_by_Elena_Ostasheva.jpg'"
                        ),
                        "shell": True,
                    },
                }
            ],
        },
        {
            "name": "cursor_size",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": "gsettings set org.gnome.desktop.interface cursor-size 16",
                        "shell": True,
                    },
                }
            ],
        },
        {
            "name": "dock_position",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": "gsettings set org.gnome.shell.extensions.dash-to-dock dock-position 'RIGHT'",
                        "shell": True,
                    },
                }
            ],
        },
        {
            "name": "icon_theme",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": "gsettings set org.gnome.desktop.interface icon-theme 'Humanity'",
                        "shell": True,
                    },
                }
            ],
        },
        {
            "name": "timezone",
            "config": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": "echo '{CLIENT_PASSWORD}' | sudo -S timedatectl set-timezone Europe/London",
                        "shell": True,
                    },
                }
            ],
        },
    ],
}


def get_perturbation_configs(perturbation_set: str = "1"):
    """Return all perturbation config entries for the given set, flattened into a single list."""
    if perturbation_set not in PERTURBATION_SETS:
        raise ValueError(f"Unknown perturbation set '{perturbation_set}'. Available: {list(PERTURBATION_SETS.keys())}")
    configs = []
    for p in PERTURBATION_SETS[perturbation_set]:
        configs.extend(p["config"])
    return configs
