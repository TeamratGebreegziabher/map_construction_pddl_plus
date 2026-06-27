(define (domain osm-urban-navigation-pddlplus)

  (:requirements
    :typing
    :negative-preconditions
    :numeric-fluents
    :processes
    :events
  )

  (:types
    vehicle
    location
  )

  (:predicates
    ; vehicle position and motion
    (at ?v - vehicle ?l - location)
    (moving ?v - vehicle)
    (moving-from ?v - vehicle ?l - location)
    (moving-to ?v - vehicle ?l - location)
    (visited ?v - vehicle ?l - location)
    ; road graph
    (connected ?from ?to - location)
    (blocked ?from ?to - location)
    ; charging
    (charging-station ?l - location)
    (charging ?v - vehicle)
    ; traffic signals — three-phase: green -> yellow -> red -> green
    (has-signal ?l - location)
    (signal-green ?l - location)
    (signal-yellow ?l - location)
    (signal-red ?l - location)
    ; coordination
    (priority ?v - vehicle)
  )

  (:functions
    ; road graph
    (road-distance ?from ?to - location)
    ; vehicle state
    (remaining-distance ?v - vehicle)
    (speed ?v - vehicle)
    (battery ?v - vehicle)
    (max-battery ?v - vehicle)
    (battery-rate ?v - vehicle)
    (battery-consumption-per-meter ?v - vehicle)
    (charge-rate ?v - vehicle)
    ; charging infrastructure
    (station-capacity ?l - location)
    (station-load ?l - location)
    ; signal timing
    (signal-timer ?l - location)
    (green-duration ?l - location)
    (yellow-duration ?l - location)
    (red-duration ?l - location)
    ; global metrics
    (total-distance)
    (total-time)
  )

  ; -------------------------------------------------------------------------
  ; Actions
  ; -------------------------------------------------------------------------

  (:action start-move
    :parameters (?v - vehicle ?from ?to - location)
    :precondition
      (and
        (at ?v ?from)
        (connected ?from ?to)
        (not (blocked ?from ?to))
        (not (moving ?v))
        (not (charging ?v))
        (not (visited ?v ?to))
        ; traffic signals — priority vehicles bypass red and yellow
        (or (priority ?v) (not (signal-red ?from)))
        (or (priority ?v) (not (signal-yellow ?from)))
        ; battery check: must have enough charge to traverse this edge
        (>= (battery ?v)
            (* (road-distance ?from ?to)
               (battery-consumption-per-meter ?v)))
      )
    :effect
      (and
        (not (at ?v ?from))
        (moving ?v)
        (moving-from ?v ?from)
        (moving-to ?v ?to)
        (assign (remaining-distance ?v) (road-distance ?from ?to))
        (increase (total-distance) (road-distance ?from ?to))
      )
  )

  (:action charge
    :parameters (?v - vehicle ?l - location)
    :precondition
      (and
        (at ?v ?l)
        (charging-station ?l)
        (not (moving ?v))
        (not (charging ?v))
        (< (battery ?v) (max-battery ?v))
        (< (station-load ?l) (station-capacity ?l))
      )
    :effect
      (and
        (charging ?v)
        (increase (station-load ?l) 1)
      )
  )

  ; -------------------------------------------------------------------------
  ; Processes
  ; -------------------------------------------------------------------------

  ; Travelling — vehicle moves at its configured speed.
  ; Congested edges are handled at problem generation time by inflating
  ; road-distance, so the planner naturally avoids or penalises them
  ; without requiring any extra fluent or formula here.
  (:process travelling
    :parameters (?v - vehicle ?from ?to - location)
    :precondition
      (and
        (moving ?v)
        (moving-from ?v ?from)
        (moving-to ?v ?to)
      )
    :effect
      (and
        (decrease (remaining-distance ?v)
          (* #t (speed ?v)))
        (decrease (battery ?v)
          (* #t (battery-rate ?v)))
        (increase (total-time) (* #t 1))
      )
  )

  (:process charging-process
    :parameters (?v - vehicle)
    :precondition
      (charging ?v)
    :effect
      (increase (battery ?v) (* #t (charge-rate ?v)))
  )

  (:process signal-cycling
    :parameters (?l - location)
    :precondition
      (has-signal ?l)
    :effect
      (decrease (signal-timer ?l) (* #t 1))
  )

  ; -------------------------------------------------------------------------
  ; Events
  ; -------------------------------------------------------------------------

  (:event arrive
    :parameters (?v - vehicle ?from ?to - location)
    :precondition
      (and
        (moving ?v)
        (moving-from ?v ?from)
        (moving-to ?v ?to)
        (<= (remaining-distance ?v) 0)
      )
    :effect
      (and
        (not (moving ?v))
        (not (moving-from ?v ?from))
        (not (moving-to ?v ?to))
        (at ?v ?to)
        (assign (remaining-distance ?v) 0)
        (visited ?v ?to)
      )
  )

  (:event fully-charged
    :parameters (?v - vehicle ?l - location)
    :precondition
      (and
        (charging ?v)
        (at ?v ?l)
        (charging-station ?l)
        (>= (battery ?v) (max-battery ?v))
      )
    :effect
      (and
        (not (charging ?v))
        (assign (battery ?v) (max-battery ?v))
        (decrease (station-load ?l) 1)
      )
  )

  ; Signal phase: green -> yellow
  (:event signal-switch-to-yellow
    :parameters (?l - location)
    :precondition
      (and
        (has-signal ?l)
        (signal-green ?l)
        (<= (signal-timer ?l) 0)
      )
    :effect
      (and
        (not (signal-green ?l))
        (signal-yellow ?l)
        (assign (signal-timer ?l) (yellow-duration ?l))
      )
  )

  ; Signal phase: yellow -> red
  (:event signal-switch-to-red
    :parameters (?l - location)
    :precondition
      (and
        (has-signal ?l)
        (signal-yellow ?l)
        (<= (signal-timer ?l) 0)
      )
    :effect
      (and
        (not (signal-yellow ?l))
        (signal-red ?l)
        (assign (signal-timer ?l) (red-duration ?l))
      )
  )

  ; Signal phase: red -> green
  (:event signal-switch-to-green
    :parameters (?l - location)
    :precondition
      (and
        (has-signal ?l)
        (signal-red ?l)
        (<= (signal-timer ?l) 0)
      )
    :effect
      (and
        (not (signal-red ?l))
        (signal-green ?l)
        (assign (signal-timer ?l) (green-duration ?l))
      )
  )

  ; Emergency preemption — a priority vehicle approaching a signalised node
  ; forces it immediately to green, regardless of current phase.
  (:event signal-preempt-green
    :parameters (?v - vehicle ?l - location)
    :precondition
      (and
        (priority ?v)
        (moving-to ?v ?l)
        (has-signal ?l)
        (not (signal-green ?l))
      )
    :effect
      (and
        (not (signal-red ?l))
        (not (signal-yellow ?l))
        (signal-green ?l)
        (assign (signal-timer ?l) (green-duration ?l))
      )
  )
)
