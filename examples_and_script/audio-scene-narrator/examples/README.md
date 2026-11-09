# Example clips

Ten-second scenes with a steady background and 2-3 foreground sounds, several of which
overlap. Pick them in the app under **Example**. Regenerate them with
`python scripts/make_examples.py <path-to-ESC-50> examples`.

| clip | scene | foreground events (s) | overlap |
|---|---|---|---|
| `01_rainy_street.wav` | A car horn honks while a siren passes in the rain. | car_horn 2.0-5.5, siren 3.0-6.5 | car_horn + siren (2.5 s) |
| `02_farm_morning.wav` | A rooster crows over a mooing cow, then a dog barks, all in the wind. | rooster 1.0-3.0, cow 2.2-5.7, dog 6.5-9.8 | rooster + cow (0.8 s) |
| `03_busy_home.wav` | A baby cries while someone knocks on the door, over a vacuum cleaner. | crying_baby 1.7-4.9, door_wood_knock 3.0-6.5 | crying_baby + door_wood_knock (1.9 s) |
| `04_park_afternoon.wav` | A dog barks and people laugh at the same time, with birds chirping. | laughing 2.0-5.5, dog 2.8-6.3, clapping 6.6-9.8 | laughing + dog (2.7 s) |
| `05_night_outside.wav` | A crow caws and a cat meows over each other, with crickets. | crow 2.5-6.0, cat 3.5-6.7 | crow + cat (2.5 s) |
| `06_fireside.wav` | Someone coughs, an alarm clock rings and they sneeze, by a crackling fire. | coughing 1.5-3.9, clock_alarm 2.5-6.0, sneezing 7.0-9.8 | coughing + clock_alarm (1.4 s) |
| `07_train_ride.wav` | A baby cries and people laugh on a moving train. | crying_baby 2.0-5.5, laughing 3.0-6.5 | crying_baby + laughing (2.5 s) |
| `08_hard_quiet_siren.wav` | HARD: a distant siren barely above heavy rain, with a loud knock. | siren 1.0-4.3, door_wood_knock 5.0-8.5 | none |

## Credits

The clips in this folder are mixed from recordings in the ESC-50 dataset
(K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015,
https://github.com/karolpiczak/ESC-50). ESC-50 is licensed under
[CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/); the recordings were
originally published on [Freesound](https://freesound.org). These mixes are shared under
the same license, for non-commercial use only.

Ground truth (event times, roles, levels and overlaps) is in `scenes.json`.

| ESC-50 file | Freesound sound |
|---|---|
| 1-17565-A-12.wav | https://freesound.org/s/17565/ |
| 1-187207-A-20.wav | https://freesound.org/s/187207/ |
| 1-24074-A-43.wav | https://freesound.org/s/24074/ |
| 1-27724-A-1.wav | https://freesound.org/s/27724/ |
| 1-54918-A-14.wav | https://freesound.org/s/54918/ |
| 2-114587-A-0.wav | https://freesound.org/s/114587/ |
| 2-117271-A-0.wav | https://freesound.org/s/117271/ |
| 2-141682-B-36.wav | https://freesound.org/s/141682/ |
| 2-262579-A-45.wav | https://freesound.org/s/262579/ |
| 2-50666-A-20.wav | https://freesound.org/s/50666/ |
| 2-70936-A-42.wav | https://freesound.org/s/70936/ |
| 2-70939-A-42.wav | https://freesound.org/s/70939/ |
| 3-120526-A-37.wav | https://freesound.org/s/120526/ |
| 3-142005-A-10.wav | https://freesound.org/s/142005/ |
| 3-152997-A-26.wav | https://freesound.org/s/152997/ |
| 3-180147-A-30.wav | https://freesound.org/s/180147/ |
| 4-133047-A-5.wav | https://freesound.org/s/133047/ |
| 4-152995-A-24.wav | https://freesound.org/s/152995/ |
| 4-157297-A-21.wav | https://freesound.org/s/157297/ |
| 4-171823-A-13.wav | https://freesound.org/s/171823/ |
| 4-173865-A-9.wav | https://freesound.org/s/173865/ |
| 4-186518-A-30.wav | https://freesound.org/s/186518/ |
| 5-117773-A-16.wav | https://freesound.org/s/117773/ |
| 5-198321-A-10.wav | https://freesound.org/s/198321/ |
| 5-202795-A-3.wav | https://freesound.org/s/202795/ |
| 5-221950-A-22.wav | https://freesound.org/s/221950/ |
| 5-242932-A-26.wav | https://freesound.org/s/242932/ |
