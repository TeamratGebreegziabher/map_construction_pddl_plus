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
    (blocked ?from ?to - location)
    (visited ?l - location)
    (connected ?from ?to - location)
    (moving ?v - vehicle)
    (moving-to ?v - vehicle ?to - location)
  )
  (:functions
    (road-distance ?from ?to - location)
    (speed ?v - vehicle)
    (battery ?v - vehicle)
    (battery-rate ?v - vehicle)
    (battery-consumption-per-meter ?v - vehicle)
    (remaining-distance ?v - vehicle)
    (total-distance)
    (total-time)
  )

  (:action start-move
    :parameters (?v - vehicle ?from ?to - location)
    :precondition
      (and
        (at ?v ?from)
        (connected ?from ?to)
        (not (blocked ?from ?to))
        (not (moving ?v))
        (not (visited ?to))
        (>= (battery ?v)
            (* (road-distance ?from ?to)
               (battery-consumption-per-meter ?v)))
      )
    :effect
      (and
        (not (at ?v ?from))
        (moving ?v)
        (moving-to ?v ?to)
        (assign (remaining-distance ?v)
                (road-distance ?from ?to))
        (increase (total-distance)
                  (road-distance ?from ?to))
      )
  )

  (:process travelling
    :parameters (?v - vehicle)
    :precondition
      (moving ?v)
    :effect
      (and
        (decrease (remaining-distance ?v)
                  (* #t (speed ?v)))
        (decrease (battery ?v)
                  (* #t (battery-rate ?v)))
        (increase (total-time)
                  (* #t 1))
      )
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
        (visited ?to)
      )
  )
)
