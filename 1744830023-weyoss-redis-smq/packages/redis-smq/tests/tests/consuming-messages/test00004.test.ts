/*
 * Copyright (c)
 * Weyoss <weyoss@protonmail.com>
 * https://github.com/weyoss
 *
 * This source code is licensed under the MIT license found in the LICENSE file
 * in the root directory of this source tree.
 */

import { expect, vitest, test } from 'vitest';
import bluebird from 'bluebird';
import { ProducibleMessage } from '../../../src/lib/index.js';
import { getConsumer } from '../../common/consumer.js';
import { getEventBus } from '../../common/event-bus-redis.js';
import { untilMessageDeadLettered } from '../../common/events.js';
import {
  createQueue,
  getDefaultQueue,
} from '../../common/message-producing-consuming.js';
import { getProducer } from '../../common/producer.js';
import { getQueueDeadLetteredMessages } from '../../common/queue-dead-lettered-messages.js';

test('A message is dead-lettered and not delivered when messageTTL is exceeded', async () => {
  const eventBus = await getEventBus();
  const defaultQueue = getDefaultQueue();
  await createQueue(defaultQueue, false);

  const producer = getProducer();
  await producer.runAsync();

  const consumer = getConsumer();
  const consume = vitest.spyOn(consumer, 'consume');

  let unacknowledged = 0;
  eventBus.on('consumer.consumeMessage.messageUnacknowledged', () => {
    unacknowledged += 1;
  });

  const msg = new ProducibleMessage();
  msg.setBody({ hello: 'world' }).setTTL(3000).setQueue(getDefaultQueue());

  const [id] = await producer.produceAsync(msg);
  await bluebird.delay(5000);
  consumer.run(() => void 0);

  await untilMessageDeadLettered(consumer);
  expect(consume).toHaveBeenCalledTimes(0);
  expect(unacknowledged).toBe(1);
  const deadLetteredMessages = await getQueueDeadLetteredMessages();
  const list = await deadLetteredMessages.getMessagesAsync(
    defaultQueue,
    0,
    100,
  );
  expect(list.totalItems).toBe(1);
  expect(list.items[0].id).toBe(id);
});
