from event.base import IntrusionProducerBase, CameraSubscriberBase


def make_intrusion_producer() -> IntrusionProducerBase:
    from event.kafka_intrusion_producer import KafkaIntrusionProducer
    return KafkaIntrusionProducer()


def make_camera_subscriber(camera_manager) -> CameraSubscriberBase:
    from event.kafka_modify_zone_consumer import KafkaModifyZoneConsumer
    return KafkaModifyZoneConsumer(camera_manager)
