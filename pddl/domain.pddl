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
    (at ?v - vehicle ?l - location)
    (moving ?v - vehicle)
    (moving-to ?v - vehicle ?l - location)
    (visited ?v - vehicle ?l - location)
    (connected ?from ?to - location)
    (blocked ?from ?to - location)
    (charging-station ?l - location)
    (charging ?v - vehicle)
    (has-signal ?l - location)
    (signal-red ?l - location)
    (signal-green ?l - location)
    (congested ?from ?to - location)
  )

  (:functions
    (road-distance ?from ?to - location)
    (remaining-distance ?v - vehicle)
    (total-distance)
    (total-time)
    (speed ?v - vehicle)
    (battery ?v - vehicle)
    (max-battery ?v - vehicle)
    (battery-rate ?v - vehicle)
    (battery-consumption-per-meter ?v - vehicle)
    (charge-rate ?v - vehicle)
    (signal-timer ?l - location)
    (red-duration ?l - location)
    (green-duration ?l - location)
  )

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
        (not (signal-red ?from))
        (>= (battery ?v)
            (* (road-distance ?from ?to)
               (battery-consumption-per-meter ?v)))
      )
    :effect
      (and
        (not (at ?v ?from))
        (moving ?v)
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
      )
    :effect
      (charging ?v)
  )

  (:action stop-charging
    :parameters (?v - vehicle ?l - location)
    :precondition
      (and
        (at ?v ?l)
        (charging ?v)
        (not (moving ?v))
      )
    :effect
      (not (charging ?v))
  )

  (:process travelling
    :parameters (?v - vehicle)
    :precondition
      (moving ?v)
    :effect
      (and
        (decrease (remaining-distance ?v) (* #t (speed ?v)))
        (decrease (battery ?v) (* #t (battery-rate ?v)))
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

  (:event arrive
    :parameters (?v - vehicle ?to - location)
    :precondition
      (and
        (moving ?v)
        (moving-to ?v ?to)
        (<= (remaining-distance ?v) 0)
      )
    :effect
      (and
        (not (moving ?v))
        (not (moving-to ?v ?to))
        (at ?v ?to)
        (assign (remaining-distance ?v) 0)
        (visited ?v ?to)
      )
  )

  (:event fully-charged
    :parameters (?v - vehicle)
    :precondition
      (and
        (charging ?v)
        (>= (battery ?v) (max-battery ?v))
      )
    :effect
      (and
        (not (charging ?v))
        (assign (battery ?v) (max-battery ?v))
      )
  )

  (:event signal-switch-to-red
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
        (signal-red ?l)
        (assign (signal-timer ?l) (red-duration ?l))
      )
  )

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
)
