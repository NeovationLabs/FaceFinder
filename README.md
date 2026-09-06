# Locating a Person Across Multiple Security Cameras Using Face Recognition

## Overview

This project demonstrates a system that can find a specific person across
multiple security camera feeds at once. Instead of a security team manually
scrubbing through hours of footage from several cameras to find where
someone went, the system does it automatically: give it a photo of the
person, and it searches every connected camera in real time, telling you
exactly which camera saw them, when, and with what confidence.

## The Problem

Most buildings, campuses, and retail spaces have several security cameras
covering different areas — entrances, hallways, checkout counters, parking
lots. When something happens (a lost visitor, a flagged individual, an
incident that needs review), security staff currently have to:

- Manually review footage from each camera separately
- Watch long stretches of video to spot one person
- Rely on memory or vague descriptions ("guy in a blue jacket") to search

This doesn't scale. More cameras means more footage, and a human reviewer
can only watch one feed at a time.

## The Solution

This system treats "finding a person across cameras" as a search problem,
not a manual review problem:

1. **Upload a reference photo** of the person you're looking for.
2. The system encodes that person's face into a digital "faceprint" — a
   unique numerical signature that represents their facial features.
3. **Every connected camera is searched simultaneously**, in real time. As
   people appear in each feed, their faces are compared against the
   faceprint.
4. **The moment a match is found**, the system reports which camera,
   the exact time, a confidence score, and a snapshot — instantly, instead
   of hours of manual searching.

## Where This Is Useful

- **Security operations** — locating a person of interest across a large
  property (a mall, campus, office building, or airport) without needing to
  check each camera by hand.
- **Missing person search** — quickly finding a lost child, elderly relative,
  or patient across a large venue's camera network.
- **Loss prevention** — identifying a previously flagged individual the
  moment they re-enter any monitored area.
- **Access control review** — confirming where and when a specific person
  moved through a monitored space.

## What Makes This Different From a Single Camera System

A single camera watching one door can only tell you "someone matching this
description walked through." This system correlates identity **across
multiple, independent camera views** — so it can answer "where is this
person right now, across the entire property," not just "did they pass this
one spot."

## Planned Enhancements

The current version requires a reference **photo** of the person to search
for. The next phase of this project extends the search to plain-language
descriptions instead of requiring a photo at all — for example, a security
operator could simply type:

> *"Find the man in a red jacket carrying a black backpack"*

and the system would search all camera feeds for a person matching that
description, without needing an enrolled photo first. This makes the system
usable in situations where no photo of the person exists yet — for example,
searching for someone based only on a witness description.

Further planned capabilities include:
- Tracking a person's full movement path across a facility over time, not
  just single-camera detections
- Searching historical recorded footage, not only live feeds
- Alerting security staff automatically the instant a match occurs, rather
  than requiring someone to be watching

## Responsible Use

This system identifies and tracks a specific named individual across
physical space, which is a meaningfully more sensitive capability than
generic security monitoring. Any real deployment of this system should:

- Only be used on camera systems the operator owns or is authorized to
  monitor
- Include clearly posted notice that facial recognition is in use, in line
  with local law (many jurisdictions specifically require this)
- Comply with applicable biometric privacy regulations, which in several
  regions require documented consent or a specific lawful basis before
  storing or matching facial data
- Keep a human reviewer in the loop for any decision made based on a match —
  the system is designed to flag and assist, not to make automated decisions
  about individuals

## Summary

This project shows that "finding someone across a security camera network"
doesn't have to mean manually watching every feed — it can be an instant,
automated search, with the long-term goal of making that search as simple as
describing the person in plain language.